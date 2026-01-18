import argparse
import json
import os
from pathlib import Path
import requests
import torch
from transformers import JukeboxConfig, JukeboxModel
from transformers.utils import logging
def fix_jukebox_keys(state_dict, model_state_dict, key_prefix, mapping):
    new_dict = {}
    import re
    re_encoder_block_conv_in = re.compile('encoders.(\\d*).level_blocks.(\\d*).model.(\\d*).(\\d).(bias|weight)')
    re_encoder_block_resnet = re.compile('encoders.(\\d*).level_blocks.(\\d*).model.(\\d*).(\\d).model.(\\d*).model.(\\d*).(bias|weight)')
    re_encoder_block_proj_out = re.compile('encoders.(\\d*).level_blocks.(\\d*).model.(\\d*).(bias|weight)')
    re_decoder_block_conv_out = re.compile('decoders.(\\d*).level_blocks.(\\d*).model.(\\d*).(\\d).(bias|weight)')
    re_decoder_block_resnet = re.compile('decoders.(\\d*).level_blocks.(\\d*).model.(\\d*).(\\d).model.(\\d*).model.(\\d*).(bias|weight)')
    re_decoder_block_proj_in = re.compile('decoders.(\\d*).level_blocks.(\\d*).model.(\\d*).(bias|weight)')
    re_prior_cond_conv_out = re.compile('conditioner_blocks.(\\d*).cond.model.(\\d*).(\\d).(bias|weight)')
    re_prior_cond_resnet = re.compile('conditioner_blocks.(\\d*).cond.model.(\\d*).(\\d).model.(\\d*).model.(\\d*).(bias|weight)')
    re_prior_cond_proj_in = re.compile('conditioner_blocks.(\\d*).cond.model.(\\d*).(bias|weight)')
    for original_key, value in state_dict.items():
        if re_encoder_block_conv_in.fullmatch(original_key):
            regex_match = re_encoder_block_conv_in.match(original_key)
            groups = regex_match.groups()
            block_index = int(groups[2]) * 2 + int(groups[3])
            re_new_key = f'encoders.{groups[0]}.level_blocks.{groups[1]}.downsample_block.{block_index}.{groups[-1]}'
            key = re_encoder_block_conv_in.sub(re_new_key, original_key)
        elif re_encoder_block_resnet.fullmatch(original_key):
            regex_match = re_encoder_block_resnet.match(original_key)
            groups = regex_match.groups()
            block_index = int(groups[2]) * 2 + int(groups[3])
            conv_index = {'1': 1, '3': 2}[groups[-2]]
            prefix = f'encoders.{groups[0]}.level_blocks.{groups[1]}.downsample_block.{block_index}.'
            resnet_block = f'resnet_block.{groups[-3]}.conv1d_{conv_index}.{groups[-1]}'
            re_new_key = prefix + resnet_block
            key = re_encoder_block_resnet.sub(re_new_key, original_key)
        elif re_encoder_block_proj_out.fullmatch(original_key):
            regex_match = re_encoder_block_proj_out.match(original_key)
            groups = regex_match.groups()
            re_new_key = f'encoders.{groups[0]}.level_blocks.{groups[1]}.proj_out.{groups[-1]}'
            key = re_encoder_block_proj_out.sub(re_new_key, original_key)
        elif re_decoder_block_conv_out.fullmatch(original_key):
            regex_match = re_decoder_block_conv_out.match(original_key)
            groups = regex_match.groups()
            block_index = int(groups[2]) * 2 + int(groups[3]) - 2
            re_new_key = f'decoders.{groups[0]}.level_blocks.{groups[1]}.upsample_block.{block_index}.{groups[-1]}'
            key = re_decoder_block_conv_out.sub(re_new_key, original_key)
        elif re_decoder_block_resnet.fullmatch(original_key):
            regex_match = re_decoder_block_resnet.match(original_key)
            groups = regex_match.groups()
            block_index = int(groups[2]) * 2 + int(groups[3]) - 2
            conv_index = {'1': 1, '3': 2}[groups[-2]]
            prefix = f'decoders.{groups[0]}.level_blocks.{groups[1]}.upsample_block.{block_index}.'
            resnet_block = f'resnet_block.{groups[-3]}.conv1d_{conv_index}.{groups[-1]}'
            re_new_key = prefix + resnet_block
            key = re_decoder_block_resnet.sub(re_new_key, original_key)
        elif re_decoder_block_proj_in.fullmatch(original_key):
            regex_match = re_decoder_block_proj_in.match(original_key)
            groups = regex_match.groups()
            re_new_key = f'decoders.{groups[0]}.level_blocks.{groups[1]}.proj_in.{groups[-1]}'
            key = re_decoder_block_proj_in.sub(re_new_key, original_key)
        elif re_prior_cond_conv_out.fullmatch(original_key):
            regex_match = re_prior_cond_conv_out.match(original_key)
            groups = regex_match.groups()
            block_index = int(groups[1]) * 2 + int(groups[2]) - 2
            re_new_key = f'conditioner_blocks.upsampler.upsample_block.{block_index}.{groups[-1]}'
            key = re_prior_cond_conv_out.sub(re_new_key, original_key)
        elif re_prior_cond_resnet.fullmatch(original_key):
            regex_match = re_prior_cond_resnet.match(original_key)
            groups = regex_match.groups()
            block_index = int(groups[1]) * 2 + int(groups[2]) - 2
            conv_index = {'1': 1, '3': 2}[groups[-2]]
            prefix = f'conditioner_blocks.upsampler.upsample_block.{block_index}.'
            resnet_block = f'resnet_block.{groups[-3]}.conv1d_{conv_index}.{groups[-1]}'
            re_new_key = prefix + resnet_block
            key = re_prior_cond_resnet.sub(re_new_key, original_key)
        elif re_prior_cond_proj_in.fullmatch(original_key):
            regex_match = re_prior_cond_proj_in.match(original_key)
            groups = regex_match.groups()
            re_new_key = f'conditioner_blocks.upsampler.proj_in.{groups[-1]}'
            key = re_prior_cond_proj_in.sub(re_new_key, original_key)
        else:
            key = original_key
        key = replace_key(key)
        if f'{key_prefix}.{key}' not in model_state_dict or key is None:
            print(f'failed converting {original_key} to {key}, does not match')
        elif value.shape != model_state_dict[f'{key_prefix}.{key}'].shape:
            val = model_state_dict[f'{key_prefix}.{key}']
            print(f'{original_key}-> {key} : \nshape {val.shape} and {value.shape}, do not match')
            key = original_key
        mapping[key] = original_key
        new_dict[key] = value
    return new_dict