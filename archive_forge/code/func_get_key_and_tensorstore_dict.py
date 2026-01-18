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
def get_key_and_tensorstore_dict(layer, checkpoint_info, switch_checkpoint_path):
    if 'metadata' in layer:
        split_layer = layer.split('metadata')
        curr_real_layer_name = ''.join(split_layer[0])[:-1]
        split_layer = [tuple(('metadata' + split_layer[1]).split('/'))]
    elif 'kvstore' in layer:
        split_layer = layer.split('kvstore')
        curr_real_layer_name = ''.join(split_layer[0])[:-1]
        split_layer = [tuple(('kvstore' + split_layer[1]).split('/'))]
    else:
        split_layer = layer.split('/')
        curr_real_layer_name = '/'.join(split_layer[:-1])
        split_layer[-1] = (split_layer[-1],)
    if 'kvstore/path' in layer:
        content = f'{switch_checkpoint_path}/{checkpoint_info[layer]}'
    elif 'kvstore/driver' in layer:
        content = 'file'
    else:
        content = checkpoint_info[layer]
    return (curr_real_layer_name, split_layer, content)