import argparse
import torch
from transformers import UnivNetConfig, UnivNetModel, logging
def get_key_mapping(config: UnivNetConfig):
    mapping = {}
    for i in range(len(config.resblock_stride_sizes)):
        mapping[f'res_stack.{i}.convt_pre.1.weight_g'] = f'resblocks.{i}.convt_pre.weight_g'
        mapping[f'res_stack.{i}.convt_pre.1.weight_v'] = f'resblocks.{i}.convt_pre.weight_v'
        mapping[f'res_stack.{i}.convt_pre.1.bias'] = f'resblocks.{i}.convt_pre.bias'
        kernel_predictor_mapping = get_kernel_predictor_key_mapping(config, old_prefix=f'res_stack.{i}.kernel_predictor', new_prefix=f'resblocks.{i}.kernel_predictor')
        mapping.update(kernel_predictor_mapping)
        for j in range(len(config.resblock_dilation_sizes[i])):
            mapping[f'res_stack.{i}.conv_blocks.{j}.1.weight_g'] = f'resblocks.{i}.resblocks.{j}.conv.weight_g'
            mapping[f'res_stack.{i}.conv_blocks.{j}.1.weight_v'] = f'resblocks.{i}.resblocks.{j}.conv.weight_v'
            mapping[f'res_stack.{i}.conv_blocks.{j}.1.bias'] = f'resblocks.{i}.resblocks.{j}.conv.bias'
    mapping['conv_post.1.weight_g'] = 'conv_post.weight_g'
    mapping['conv_post.1.weight_v'] = 'conv_post.weight_v'
    mapping['conv_post.1.bias'] = 'conv_post.bias'
    return mapping