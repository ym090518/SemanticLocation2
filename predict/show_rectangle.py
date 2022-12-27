# **
# * Copyright @2022 AI, AIRCAS. (mails.ucas.ac.cn)
#
# @author yuanzhiqiang <yuanzhiqiang19@mails.ucas.ac.cn>
#         2022/05/03

import json
import os
import sys
import time

import cv2
import numpy as np


from PIL import ImageDraw,Image

import utils
from model_encoder import Encoder
from model_init import model_init

sys.path.append("..")
from evaluations.SLM import SLM

# 将图片按照256*256剪裁，并返回剪裁时间和剪切次数
# img_path：图片位置，比如0.jpg
# subimages_dir：剪裁的图片放的位置
# cut_count：剪裁的次数
def split_image(img_path, subimages_dir):
    # Read Image
    source_img = cv2.imread(img_path)
    img_height = np.shape(source_img)[0]
    img_weight = np.shape(source_img)[1]
    step = 512
    logger.info("cut_img size:{}x{}".format(step, step))
    # 剪裁图像
    t1 = time.time()
    # Cut img
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    coords = []
    id = 0
    for h in range(0, img_height, step):
        id = id + 1
        h_start, h_end = h, h + step
        if (h_end >= img_height):
            h_start, h_end = img_height - step, img_height
        for w in range(0, img_weight, step):
            w_start, w_end = w, w + step
            # bound?
            if w_end >= img_weight:
                w_start, w_end = img_weight - step, img_weight
            cut_img_name = str(h_start) + "_" + str(h_end) + "_" + str(w_start) + "_" + str(w_end) + ".jpg"
            print(cut_img_name)
            cut_img = source_img[h_start:h_end, w_start:w_end]
            cut_img = cv2.resize(cut_img, (256, 256), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(subimages_dir, cut_img_name), cut_img)
            coords.append([int(w_start), int(h_start), int(w_end), int(h_end)])


    color = [(0, 255, 0)]
    for coord in coords:
        x1 = coord[0]
        y1 = coord[1]
        x2 = coord[2]
        y2 = coord[3]
        draw.rectangle([x1, y1, x2, y2], outline=color[0], width=3,fill=255)
    img.show()


    split_time = time.time() - t1
    logger.info("Image {} has been split successfully.".format(img_path))
    logger.info("cut image cost {}s.".format(split_time))

    return split_time

# 获取裁剪图像的特征向量
# subimages_dir：剪裁好的图片所在位置，比如：0_subimages
# img_vectors：每一张剪裁好图片的特征向量的列表集合
# text：文本
def show_sim_rectangles(subimages_dir, text, encoder,img_path,count):
    # read subimages
    subimages = os.listdir(subimages_dir)

    # text vector
    text_vector = encoder.text_encoder(model, vocab, text)

    # 获取每一个剪裁好图片的特征,并计算与文本的相似度
    sim_results = []
    t1 = time.time()
    for subimage in subimages:
        count = count+1
        # 一维的特征向量
        image_vector = encoder.image_encoder(model, os.path.join(subimages_dir, subimage))
        sim_results.append(encoder.cosine_sim(text_vector, image_vector))
    img_vector_time = time.time()-t1
    logger.info("image_vector have been extracted,cost {}s ...".format(img_vector_time))
    print("没有排序前的sim_results")
    print(sim_results)


    # 将相似度排序，返回索引列表
    sim_index_sort = np.argsort(sim_results)
    print("按照相似度排序后的sim_results的索引（从小到大）：")
    print(sim_index_sort)
    select_sim_nums = int(len(sim_results) * 0.15) + 1
    print('选择了{}张图片'.format(select_sim_nums))
    sim_results_selected = sim_index_sort[-select_sim_nums:]
    source_img = Image.open(img_path)
    draw = ImageDraw.Draw(source_img)
    coords = []
    img = cv2.imread(img_path)
    img_height = np.shape(img)[0]
    img_weight = np.shape(img)[1]
    for index in sim_results_selected:
        image_name = subimages[index]
        print("图像{}的相似度为：{}".format(image_name,sim_results[index]))
        h_start1, h_end1, w_start1, w_end1 = image_name.replace(".jpg", "").split("_")
        h_start, h_end, w_start, w_end = int(h_start1), int(h_end1), int(w_start1), int(w_end1)
        coords.append([w_start, h_start, w_end, h_end])

        cut_img_size = 512
        p1 = [int(h_start), int(w_start)]
        p2 = [int(h_start), int(w_end)]
        p3 = [int(h_end), int(w_start)]
        p4 = [int(h_end), int(w_end)]
        p5 = [int(h_start) + cut_img_size * 1 / 4, int(w_start) + cut_img_size * 1 / 4]
        p6 = [int(h_start) + cut_img_size * 1 / 4, int(w_start) + cut_img_size * 3 / 4]
        p7 = [int(h_start) + cut_img_size * 3 / 4, int(w_start) + cut_img_size * 1 / 4]
        p8 = [int(h_start) + cut_img_size * 3 / 4, int(w_start) + cut_img_size * 3 / 4]
        p9 = [int(h_start) + cut_img_size * 1 / 2, int(w_start) + cut_img_size * 1 / 2]

        points = [p1, p2, p3, p4, p5, p6, p7, p8, p9]
        for point in points:
            h = point[0]
            w = point[1]
            sizes = [64, 128]
            for size in sizes:
                h_start1 = int(h - size if h - size > 0 else 0)
                h_end1 = int(h_start1 + size * 2 if h_start1 + size * 2 < img_height else img_height)
                w_start1 = int(w - size if w - size > 0 else 0)
                w_end1 = int(w_start1 + size * 2 if w_start1 + size * 2 < img_weight else img_weight)
                coords.append([int(w_start1), int(h_start1), int(w_end1), int(h_end1)])
    color = [(0, 255, 0)]
    for coord in coords:
        x1 = coord[0]
        y1 = coord[1]
        x2 = coord[2]
        y2 = coord[3]
        draw.rectangle([x1, y1, x2, y2], outline=color[0], width=3)
    source_img.show()

if __name__ == "__main__":

    import argparse

    # settings
    parser = argparse.ArgumentParser(description="SLM")
    parser.add_argument("--yaml_path", type=str, default="option/RSITMD/RSITMD_AMFMN.yaml", help="config yaml path")
    parser.add_argument("--cache_path", type=str, default="cache/show_rectangle", help="cache path")
    parser.add_argument("--src_data_path", type=str, default="../test_data/imgs/split", help="testset images path")
    parser.add_argument("--src_anno_path", type=str, default="../test_data/annotations/anno.json", help="testset annotations path")
    opt = parser.parse_args()

    # mkdir
    if not os.path.exists(opt.cache_path):
        os.mkdir(opt.cache_path)

    # logging
    logger = utils.get_logger(os.path.join(opt.cache_path, 'log.txt'))

    # init model
    model, vocab = model_init(
        prefix_path = "./",
        yaml_path = opt.yaml_path
    )

    # init encoder
    encoder = Encoder(model)

    # start eval
    slm_metric = SLM()

    # load from annotations
    with open(opt.src_anno_path,'r',encoding='utf8')as fp:
            json_data = json.load(fp)

    t1_all, t2_all, t3_all, t4_all,t5_all = 0, 0, 0, 0,0
    total_time = time.time()

    for idx, item in enumerate(json_data):
        # load sample
        img = item['jpg_name']
        text = item['caption']
        points = item['points']
        count = 0
        if idx!=39:
            continue



        # path
        img_path = os.path.join(opt.src_data_path, img)
        subimage_files_dir = os.path.join(opt.cache_path, os.path.basename(img_path).split(".")[0])
        heatmap_path = os.path.join(opt.cache_path, "heatmap_{}.jpg".format(idx))
        probmap_path = os.path.join(opt.cache_path, "probmap_{}.jpg".format(idx))
        addmap_path = os.path.join(opt.cache_path, "addmap_{}.jpg".format(idx))

        # 裁切图像文件夹
        subimages_dir = subimage_files_dir + '_subimages'
        if os.path.exists(subimages_dir):
            utils.delete_dire(subimages_dir)
        else:
            os.makedirs(subimages_dir)


        # logging
        logger.info("Processing {}/{}: {}".format(idx, len(json_data), img))
        logger.info("Corresponding text: {}".format(text))

        # processing
        t1 = split_image(img_path,subimages_dir)
        show_sim_rectangles(subimages_dir,text,encoder,img_path,0)
        exit()


