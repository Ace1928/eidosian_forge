import argparse
import requests
import torch
from PIL import Image
from transformers import SwinConfig, SwinForMaskedImageModeling, ViTImageProcessor
def get_swin_config(model_name):
    config = SwinConfig(image_size=192)
    if 'base' in model_name:
        window_size = 6
        embed_dim = 128
        depths = (2, 2, 18, 2)
        num_heads = (4, 8, 16, 32)
    elif 'large' in model_name:
        window_size = 12
        embed_dim = 192
        depths = (2, 2, 18, 2)
        num_heads = (6, 12, 24, 48)
    else:
        raise ValueError('Model not supported, only supports base and large variants')
    config.window_size = window_size
    config.embed_dim = embed_dim
    config.depths = depths
    config.num_heads = num_heads
    return config