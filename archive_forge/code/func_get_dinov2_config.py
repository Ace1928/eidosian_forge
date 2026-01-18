import argparse
import json
from pathlib import Path
import requests
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms
from transformers import BitImageProcessor, Dinov2Config, Dinov2ForImageClassification, Dinov2Model
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling
from transformers.utils import logging
def get_dinov2_config(model_name, image_classifier=False):
    config = Dinov2Config(image_size=518, patch_size=14)
    if 'vits' in model_name:
        config.hidden_size = 384
        config.num_attention_heads = 6
    elif 'vitb' in model_name:
        pass
    elif 'vitl' in model_name:
        config.hidden_size = 1024
        config.num_hidden_layers = 24
        config.num_attention_heads = 16
    elif 'vitg' in model_name:
        config.use_swiglu_ffn = True
        config.hidden_size = 1536
        config.num_hidden_layers = 40
        config.num_attention_heads = 24
    else:
        raise ValueError('Model not supported')
    if image_classifier:
        repo_id = 'huggingface/label-files'
        filename = 'imagenet-1k-id2label.json'
        config.num_labels = 1000
        config.id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type='dataset'), 'r'))
        config.id2label = {int(k): v for k, v in config.id2label.items()}
    return config