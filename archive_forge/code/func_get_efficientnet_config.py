import argparse
import json
import os
import numpy as np
import PIL
import requests
import tensorflow.keras.applications.efficientnet as efficientnet
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from tensorflow.keras.preprocessing import image
from transformers import (
from transformers.utils import logging
def get_efficientnet_config(model_name):
    config = EfficientNetConfig()
    config.hidden_dim = CONFIG_MAP[model_name]['hidden_dim']
    config.width_coefficient = CONFIG_MAP[model_name]['width_coef']
    config.depth_coefficient = CONFIG_MAP[model_name]['depth_coef']
    config.image_size = CONFIG_MAP[model_name]['image_size']
    config.dropout_rate = CONFIG_MAP[model_name]['dropout_rate']
    config.depthwise_padding = CONFIG_MAP[model_name]['dw_padding']
    repo_id = 'huggingface/label-files'
    filename = 'imagenet-1k-id2label.json'
    config.num_labels = 1000
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type='dataset'), 'r'))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
    return config