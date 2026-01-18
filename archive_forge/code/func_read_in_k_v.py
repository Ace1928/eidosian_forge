import argparse
from pathlib import Path
import requests
import torch
from PIL import Image
from transformers import PvtConfig, PvtForImageClassification, PvtImageProcessor
from transformers.utils import logging
def read_in_k_v(state_dict, config):
    for i in range(config.num_encoder_blocks):
        for j in range(config.depths[i]):
            kv_weight = state_dict.pop(f'pvt.encoder.block.{i}.{j}.attention.self.kv.weight')
            kv_bias = state_dict.pop(f'pvt.encoder.block.{i}.{j}.attention.self.kv.bias')
            state_dict[f'pvt.encoder.block.{i}.{j}.attention.self.key.weight'] = kv_weight[:config.hidden_sizes[i], :]
            state_dict[f'pvt.encoder.block.{i}.{j}.attention.self.key.bias'] = kv_bias[:config.hidden_sizes[i]]
            state_dict[f'pvt.encoder.block.{i}.{j}.attention.self.value.weight'] = kv_weight[config.hidden_sizes[i]:, :]
            state_dict[f'pvt.encoder.block.{i}.{j}.attention.self.value.bias'] = kv_bias[config.hidden_sizes[i]:]