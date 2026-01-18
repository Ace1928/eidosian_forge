import argparse
import collections
import os
import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax.training import checkpoints
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import (
from transformers.utils import logging
def get_owlv2_config(model_name):
    if 'large' in model_name:
        image_size = 1008
        patch_size = 14
        vision_hidden_size = 1024
        vision_intermediate_size = 4096
        vision_num_hidden_layers = 24
        vision_num_attention_heads = 16
        projection_dim = 768
        text_hidden_size = 768
        text_intermediate_size = 3072
        text_num_attention_heads = 12
        text_num_hidden_layers = 12
    else:
        image_size = 960
        patch_size = 16
        vision_hidden_size = 768
        vision_intermediate_size = 3072
        vision_num_hidden_layers = 12
        vision_num_attention_heads = 12
        projection_dim = 512
        text_hidden_size = 512
        text_intermediate_size = 2048
        text_num_attention_heads = 8
        text_num_hidden_layers = 12
    vision_config = Owlv2VisionConfig(patch_size=patch_size, image_size=image_size, hidden_size=vision_hidden_size, num_hidden_layers=vision_num_hidden_layers, intermediate_size=vision_intermediate_size, num_attention_heads=vision_num_attention_heads)
    text_config = Owlv2TextConfig(hidden_size=text_hidden_size, intermediate_size=text_intermediate_size, num_attention_heads=text_num_attention_heads, num_hidden_layers=text_num_hidden_layers)
    config = Owlv2Config(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), projection_dim=projection_dim)
    return config