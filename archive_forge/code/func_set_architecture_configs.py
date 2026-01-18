import argparse
import json
import gdown
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import (
def set_architecture_configs(model_name, config):
    if 'small' in model_name:
        config.hidden_size = 384
        config.intermediate_size = 1536
        config.num_hidden_layers = 12
        config.num_attention_heads = 16
        config.decoder_num_hidden_layers = 12
        config.decoder_num_attention_heads = 3
        config.decoder_hidden_size = 192
        config.decoder_intermediate_size = 768
    elif 'large' in model_name:
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_hidden_layers = 24
        config.num_attention_heads = 16
        config.decoder_num_hidden_layers = 12
        config.decoder_num_attention_heads = 8
        config.decoder_hidden_size = 512
        config.decoder_intermediate_size = 2048
    elif 'huge' in model_name:
        config.hidden_size = 1280
        config.intermediate_size = 5120
        config.num_hidden_layers = 32
        config.num_attention_heads = 16
        config.decoder_num_hidden_layers = 12
        config.decoder_num_attention_heads = 8
        config.decoder_hidden_size = 640
        config.decoder_intermediate_size = 2560
    elif 'base' not in model_name:
        raise ValueError('Model name should include either "small", "base", "large", or "huge"')