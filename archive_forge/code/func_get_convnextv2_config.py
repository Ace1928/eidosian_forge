import argparse
import json
import os
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import ConvNextImageProcessor, ConvNextV2Config, ConvNextV2ForImageClassification
from transformers.image_utils import PILImageResampling
from transformers.utils import logging
def get_convnextv2_config(checkpoint_url):
    config = ConvNextV2Config()
    if 'atto' in checkpoint_url:
        depths = [2, 2, 6, 2]
        hidden_sizes = [40, 80, 160, 320]
    if 'femto' in checkpoint_url:
        depths = [2, 2, 6, 2]
        hidden_sizes = [48, 96, 192, 384]
    if 'pico' in checkpoint_url:
        depths = [2, 2, 6, 2]
        hidden_sizes = [64, 128, 256, 512]
    if 'nano' in checkpoint_url:
        depths = [2, 2, 8, 2]
        hidden_sizes = [80, 160, 320, 640]
    if 'tiny' in checkpoint_url:
        depths = [3, 3, 9, 3]
        hidden_sizes = [96, 192, 384, 768]
    if 'base' in checkpoint_url:
        depths = [3, 3, 27, 3]
        hidden_sizes = [128, 256, 512, 1024]
    if 'large' in checkpoint_url:
        depths = [3, 3, 27, 3]
        hidden_sizes = [192, 384, 768, 1536]
    if 'huge' in checkpoint_url:
        depths = [3, 3, 27, 3]
        hidden_sizes = [352, 704, 1408, 2816]
    num_labels = 1000
    filename = 'imagenet-1k-id2label.json'
    expected_shape = (1, 1000)
    repo_id = 'huggingface/label-files'
    config.num_labels = num_labels
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type='dataset'), 'r'))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
    config.hidden_sizes = hidden_sizes
    config.depths = depths
    return (config, expected_shape)