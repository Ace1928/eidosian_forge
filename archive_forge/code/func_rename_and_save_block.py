import argparse
import json
import os
import tensorstore as ts
import torch
from flax import serialization
from flax.traverse_util import flatten_dict, unflatten_dict
from tensorflow.io import gfile
from transformers.modeling_utils import dtype_byte_size
from transformers.models.switch_transformers.convert_switch_transformers_original_flax_checkpoint_to_pytorch import (
from transformers.utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME
from transformers.utils.hub import convert_file_size_to_int
def rename_and_save_block(current_block, save_path):
    current_block = rename_keys(current_block)
    new_current_block = {}
    for k, v in current_block.items():
        new_current_block[k.replace('/', '.')] = v
    current_block = new_current_block
    torch.save(current_block, save_path)