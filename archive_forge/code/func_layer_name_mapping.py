import argparse
import json
import os
import re
import torch
from transformers import BloomConfig, BloomModel
from transformers.file_utils import CONFIG_NAME, WEIGHTS_NAME
from transformers.utils import logging
def layer_name_mapping(key, file):
    """Convert Megatron-DeepSpeed TP/PP weights mapping in transformers PP only"""
    layer_rename_map = {'word_embeddings.weight': 'word_embeddings.weight', 'word_embeddings.norm.weight': 'word_embeddings_layernorm.weight', 'word_embeddings.norm.bias': 'word_embeddings_layernorm.bias', 'weight': 'ln_f.weight', 'bias': 'ln_f.bias'}
    if key in layer_rename_map:
        return layer_rename_map[key]
    layer_number = int(re.match('.*layer_(\\d*).*', file)[1])
    layer_number -= 3
    return f'h.{layer_number}.' + key