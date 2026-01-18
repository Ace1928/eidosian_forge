import argparse
import collections
from pathlib import Path
import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download
from numpy import load
from PIL import Image
from transformers import SiglipConfig, SiglipImageProcessor, SiglipModel, SiglipProcessor, SiglipTokenizer
from transformers.utils import logging
def get_siglip_config(model_name):
    config = SiglipConfig()
    vocab_size = 250000 if 'i18n' in model_name else 32000
    image_size = model_name_to_image_size[model_name]
    patch_size = 16 if 'patch16' in model_name else 14
    config.vision_config.image_size = image_size
    config.vision_config.patch_size = patch_size
    config.text_config.vocab_size = vocab_size
    if 'base' in model_name:
        pass
    elif 'large' in model_name:
        config.text_config.hidden_size = 1024
        config.text_config.intermediate_size = 4096
        config.text_config.num_hidden_layers = 24
        config.text_config.num_attention_heads = 16
        config.vision_config.hidden_size = 1024
        config.vision_config.intermediate_size = 4096
        config.vision_config.num_hidden_layers = 24
        config.vision_config.num_attention_heads = 16
    elif 'so400m' in model_name:
        config.text_config.hidden_size = 1152
        config.text_config.intermediate_size = 4304
        config.text_config.num_hidden_layers = 27
        config.text_config.num_attention_heads = 16
        config.vision_config.hidden_size = 1152
        config.vision_config.intermediate_size = 4304
        config.vision_config.num_hidden_layers = 27
        config.vision_config.num_attention_heads = 16
    else:
        raise ValueError('Model not supported')
    return config