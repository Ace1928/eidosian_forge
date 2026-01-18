import argparse
import json
from collections import OrderedDict
from pathlib import Path
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import PoolFormerConfig, PoolFormerForImageClassification, PoolFormerImageProcessor
from transformers.utils import logging
def replace_key_with_offset(key, offset, original_name, new_name):
    """
    Replaces the key by subtracting the offset from the original layer number
    """
    to_find = original_name.split('.')[0]
    key_list = key.split('.')
    orig_block_num = int(key_list[key_list.index(to_find) - 2])
    layer_num = int(key_list[key_list.index(to_find) - 1])
    new_block_num = orig_block_num - offset
    key = key.replace(f'{orig_block_num}.{layer_num}.{original_name}', f'block.{new_block_num}.{layer_num}.{new_name}')
    return key