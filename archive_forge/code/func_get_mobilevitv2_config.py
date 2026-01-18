import argparse
import collections
import json
from pathlib import Path
import requests
import torch
import yaml
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import (
from transformers.utils import logging
def get_mobilevitv2_config(task_name, orig_cfg_file):
    config = MobileViTV2Config()
    is_segmentation_model = False
    if task_name.startswith('imagenet1k_'):
        config.num_labels = 1000
        if int(task_name.strip().split('_')[-1]) == 384:
            config.image_size = 384
        else:
            config.image_size = 256
        filename = 'imagenet-1k-id2label.json'
    elif task_name.startswith('imagenet21k_to_1k_'):
        config.num_labels = 21000
        if int(task_name.strip().split('_')[-1]) == 384:
            config.image_size = 384
        else:
            config.image_size = 256
        filename = 'imagenet-22k-id2label.json'
    elif task_name.startswith('ade20k_'):
        config.num_labels = 151
        config.image_size = 512
        filename = 'ade20k-id2label.json'
        is_segmentation_model = True
    elif task_name.startswith('voc_'):
        config.num_labels = 21
        config.image_size = 512
        filename = 'pascal-voc-id2label.json'
        is_segmentation_model = True
    orig_config = load_orig_config_file(orig_cfg_file)
    assert getattr(orig_config, 'model.classification.name', -1) == 'mobilevit_v2', 'Invalid model'
    config.width_multiplier = getattr(orig_config, 'model.classification.mitv2.width_multiplier', 1.0)
    assert getattr(orig_config, 'model.classification.mitv2.attn_norm_layer', -1) == 'layer_norm_2d', 'Norm layers other than layer_norm_2d is not supported'
    config.hidden_act = getattr(orig_config, 'model.classification.activation.name', 'swish')
    if is_segmentation_model:
        config.output_stride = getattr(orig_config, 'model.segmentation.output_stride', 16)
        if '_deeplabv3' in task_name:
            config.atrous_rates = getattr(orig_config, 'model.segmentation.deeplabv3.aspp_rates', [12, 24, 36])
            config.aspp_out_channels = getattr(orig_config, 'model.segmentation.deeplabv3.aspp_out_channels', 512)
            config.aspp_dropout_prob = getattr(orig_config, 'model.segmentation.deeplabv3.aspp_dropout', 0.1)
    repo_id = 'huggingface/label-files'
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type='dataset'), 'r'))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
    return config