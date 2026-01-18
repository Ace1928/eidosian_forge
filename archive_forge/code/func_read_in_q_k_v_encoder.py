import argparse
import requests
import torch
from PIL import Image
from torchvision import transforms as T
from transformers import (
def read_in_q_k_v_encoder(state_dict, config):
    embed_dim = config.backbone_config.embed_dim
    for layer, depth in enumerate(config.backbone_config.depths):
        hidden_size = embed_dim * 2 ** layer
        for block in range(depth):
            in_proj_weight = state_dict.pop(f'backbone.0.layers.{layer}.blocks.{block}.attn.qkv.weight')
            in_proj_bias = state_dict.pop(f'backbone.0.layers.{layer}.blocks.{block}.attn.qkv.bias')
            state_dict[f'model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.attention.self.query.weight'] = in_proj_weight[:hidden_size, :]
            state_dict[f'model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.attention.self.query.bias'] = in_proj_bias[:hidden_size]
            state_dict[f'model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.attention.self.key.weight'] = in_proj_weight[hidden_size:hidden_size * 2, :]
            state_dict[f'model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.attention.self.key.bias'] = in_proj_bias[hidden_size:hidden_size * 2]
            state_dict[f'model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.attention.self.value.weight'] = in_proj_weight[-hidden_size:, :]
            state_dict[f'model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.attention.self.value.bias'] = in_proj_bias[-hidden_size:]