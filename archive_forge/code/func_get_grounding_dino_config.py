import argparse
import requests
import torch
from PIL import Image
from torchvision import transforms as T
from transformers import (
def get_grounding_dino_config(model_name):
    if 'tiny' in model_name:
        window_size = 7
        embed_dim = 96
        depths = (2, 2, 6, 2)
        num_heads = (3, 6, 12, 24)
        image_size = 224
    elif 'base' in model_name:
        window_size = 12
        embed_dim = 128
        depths = (2, 2, 18, 2)
        num_heads = (4, 8, 16, 32)
        image_size = 384
    else:
        raise ValueError('Model not supported, only supports base and large variants')
    backbone_config = SwinConfig(window_size=window_size, image_size=image_size, embed_dim=embed_dim, depths=depths, num_heads=num_heads, out_indices=[2, 3, 4])
    config = GroundingDinoConfig(backbone_config=backbone_config)
    return config