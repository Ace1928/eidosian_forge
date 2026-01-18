import argparse
import json
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms
from transformers import BitImageProcessor, FocalNetConfig, FocalNetForImageClassification
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling
def get_focalnet_config(model_name):
    depths = [2, 2, 6, 2] if 'tiny' in model_name else [2, 2, 18, 2]
    use_conv_embed = True if 'large' in model_name or 'huge' in model_name else False
    use_post_layernorm = True if 'large' in model_name or 'huge' in model_name else False
    use_layerscale = True if 'large' in model_name or 'huge' in model_name else False
    if 'large' in model_name or 'xlarge' in model_name or 'huge' in model_name:
        if 'fl3' in model_name:
            focal_levels = [3, 3, 3, 3]
            focal_windows = [5, 5, 5, 5]
        elif 'fl4' in model_name:
            focal_levels = [4, 4, 4, 4]
            focal_windows = [3, 3, 3, 3]
    if 'tiny' in model_name or 'small' in model_name or 'base' in model_name:
        focal_windows = [3, 3, 3, 3]
        if 'lrf' in model_name:
            focal_levels = [3, 3, 3, 3]
        else:
            focal_levels = [2, 2, 2, 2]
    if 'tiny' in model_name:
        embed_dim = 96
    elif 'small' in model_name:
        embed_dim = 96
    elif 'base' in model_name:
        embed_dim = 128
    elif 'large' in model_name:
        embed_dim = 192
    elif 'xlarge' in model_name:
        embed_dim = 256
    elif 'huge' in model_name:
        embed_dim = 352
    repo_id = 'huggingface/label-files'
    if 'large' in model_name or 'huge' in model_name:
        filename = 'imagenet-22k-id2label.json'
    else:
        filename = 'imagenet-1k-id2label.json'
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type='dataset'), 'r'))
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    config = FocalNetConfig(embed_dim=embed_dim, depths=depths, focal_levels=focal_levels, focal_windows=focal_windows, use_conv_embed=use_conv_embed, id2label=id2label, label2id=label2id, use_post_layernorm=use_post_layernorm, use_layerscale=use_layerscale)
    return config