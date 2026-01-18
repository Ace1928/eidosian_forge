import argparse
import json
from pathlib import Path
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import DetrConfig, DetrForObjectDetection, DetrForSegmentation, DetrImageProcessor, ResNetConfig
from transformers.utils import logging
def get_detr_config(model_name):
    if 'resnet-50' in model_name:
        backbone_config = ResNetConfig.from_pretrained('microsoft/resnet-50')
    elif 'resnet-101' in model_name:
        backbone_config = ResNetConfig.from_pretrained('microsoft/resnet-101')
    else:
        raise ValueError('Model name should include either resnet50 or resnet101')
    config = DetrConfig(use_timm_backbone=False, backbone_config=backbone_config)
    is_panoptic = 'panoptic' in model_name
    if is_panoptic:
        config.num_labels = 250
    else:
        config.num_labels = 91
        repo_id = 'huggingface/label-files'
        filename = 'coco-detection-id2label.json'
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type='dataset'), 'r'))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}
    return (config, is_panoptic)