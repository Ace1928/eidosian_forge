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
def shard_on_the_fly(switch_checkpoint_path, dump_path, max_shard_size, dtype, weights_name: str=WEIGHTS_NAME):
    max_shard_size = convert_file_size_to_int(max_shard_size)
    sharded_state_dicts = []
    current_block = {}
    current_block_size = 0
    total_size = 0
    os.makedirs(dump_path, exist_ok=True)
    with gfile.GFile(switch_checkpoint_path + '/checkpoint', 'rb') as fp:
        checkpoint_info = serialization.msgpack_restore(fp.read())['optimizer']['target']
        checkpoint_info = flatten_dict(checkpoint_info, sep='/')
    all_layers = {}
    for layer in checkpoint_info.keys():
        curr_real_layer_name, split_layer, content = get_key_and_tensorstore_dict(layer, checkpoint_info, switch_checkpoint_path)
        if curr_real_layer_name in all_layers:
            all_layers[curr_real_layer_name][split_layer[-1]] = content
        else:
            all_layers[curr_real_layer_name] = {split_layer[-1]: content}
    for key in all_layers.keys():
        raw_weights = ts.open(unflatten_dict(all_layers[key])).result().read().result()
        raw_weights = torch.tensor(raw_weights)
        weight_size = raw_weights.numel() * dtype_byte_size(raw_weights.dtype)
        key, raw_weights = rename_base_flax_keys(tuple(key.split('/')), raw_weights)
        key = '/'.join(key)
        if current_block_size + weight_size > max_shard_size:
            save_path = os.path.join(dump_path, weights_name.replace('.bin', f'-{len(sharded_state_dicts) + 1:05d}-of-???.bin'))
            rename_and_save_block(current_block, save_path)
            sharded_state_dicts.append(current_block.keys())
            del current_block
            current_block = {}
            current_block_size = 0
        current_block[key] = raw_weights.to(getattr(torch, dtype))
        current_block_size += weight_size
        total_size += weight_size
    save_path = os.path.join(dump_path, weights_name.replace('.bin', f'-{len(sharded_state_dicts) + 1:05d}-of-???.bin'))
    rename_and_save_block(current_block, save_path)
    sharded_state_dicts.append(current_block.keys())
    if len(sharded_state_dicts) == 1:
        return ({weights_name: sharded_state_dicts[0]}, None)
    weight_map = {}
    shards = {}
    for idx, shard in enumerate(sharded_state_dicts):
        shard_file = weights_name.replace('.bin', f'-{idx + 1:05d}-of-{len(sharded_state_dicts):05d}.bin')
        temp_filename = os.path.join(dump_path, weights_name.replace('.bin', f'-{idx + 1:05d}-of-???.bin'))
        os.rename(temp_filename, os.path.join(dump_path, shard_file))
        shards[shard_file] = shard
        for key in shard:
            weight_map[key] = shard_file
    metadata = {'total_size': total_size}
    index = {'metadata': metadata, 'weight_map': weight_map}
    with open(os.path.join(dump_path, WEIGHTS_INDEX_NAME), 'w', encoding='utf-8') as f:
        content = json.dumps(index, indent=2, sort_keys=True) + '\n'
        f.write(content)
    return (metadata, index)