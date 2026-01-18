import argparse
import requests
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
from transformers import (
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
def read_in_q_v_bias(state_dict, config):
    for i in range(config.vision_config.num_hidden_layers):
        q_bias = state_dict.pop(f'visual_encoder.blocks.{i}.attn.q_bias')
        v_bias = state_dict.pop(f'visual_encoder.blocks.{i}.attn.v_bias')
        qkv_bias = torch.cat((q_bias, torch.zeros_like(v_bias, requires_grad=False), v_bias))
        state_dict[f'vision_model.encoder.layers.{i}.self_attn.qkv.bias'] = qkv_bias