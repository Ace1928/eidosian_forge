import argparse
import itertools
import math
from pathlib import Path
import requests
import torch
from PIL import Image
from torchvision import transforms
from transformers import Dinov2Config, DPTConfig, DPTForDepthEstimation, DPTImageProcessor
from transformers.utils import logging
def create_rename_keys_dpt(config):
    rename_keys = []
    for i in range(4):
        rename_keys.append((f'decode_head.reassemble_blocks.projects.{i}.conv.weight', f'neck.reassemble_stage.layers.{i}.projection.weight'))
        rename_keys.append((f'decode_head.reassemble_blocks.projects.{i}.conv.bias', f'neck.reassemble_stage.layers.{i}.projection.bias'))
        rename_keys.append((f'decode_head.reassemble_blocks.readout_projects.{i}.0.weight', f'neck.reassemble_stage.readout_projects.{i}.0.weight'))
        rename_keys.append((f'decode_head.reassemble_blocks.readout_projects.{i}.0.bias', f'neck.reassemble_stage.readout_projects.{i}.0.bias'))
        if i != 2:
            rename_keys.append((f'decode_head.reassemble_blocks.resize_layers.{i}.weight', f'neck.reassemble_stage.layers.{i}.resize.weight'))
            rename_keys.append((f'decode_head.reassemble_blocks.resize_layers.{i}.bias', f'neck.reassemble_stage.layers.{i}.resize.bias'))
    for i in range(4):
        rename_keys.append((f'decode_head.fusion_blocks.{i}.project.conv.weight', f'neck.fusion_stage.layers.{i}.projection.weight'))
        rename_keys.append((f'decode_head.fusion_blocks.{i}.project.conv.bias', f'neck.fusion_stage.layers.{i}.projection.bias'))
        if i != 0:
            rename_keys.append((f'decode_head.fusion_blocks.{i}.res_conv_unit1.conv1.conv.weight', f'neck.fusion_stage.layers.{i}.residual_layer1.convolution1.weight'))
            rename_keys.append((f'decode_head.fusion_blocks.{i}.res_conv_unit1.conv2.conv.weight', f'neck.fusion_stage.layers.{i}.residual_layer1.convolution2.weight'))
        rename_keys.append((f'decode_head.fusion_blocks.{i}.res_conv_unit2.conv1.conv.weight', f'neck.fusion_stage.layers.{i}.residual_layer2.convolution1.weight'))
        rename_keys.append((f'decode_head.fusion_blocks.{i}.res_conv_unit2.conv2.conv.weight', f'neck.fusion_stage.layers.{i}.residual_layer2.convolution2.weight'))
    for i in range(4):
        rename_keys.append((f'decode_head.convs.{i}.conv.weight', f'neck.convs.{i}.weight'))
    rename_keys.append(('decode_head.project.conv.weight', 'head.projection.weight'))
    rename_keys.append(('decode_head.project.conv.bias', 'head.projection.bias'))
    for i in range(0, 5, 2):
        rename_keys.append((f'decode_head.conv_depth.head.{i}.weight', f'head.head.{i}.weight'))
        rename_keys.append((f'decode_head.conv_depth.head.{i}.bias', f'head.head.{i}.bias'))
    return rename_keys