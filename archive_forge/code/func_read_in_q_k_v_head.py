import argparse
import collections
from pathlib import Path
import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download
from numpy import load
from PIL import Image
from transformers import SiglipConfig, SiglipImageProcessor, SiglipModel, SiglipProcessor, SiglipTokenizer
from transformers.utils import logging
def read_in_q_k_v_head(state_dict, config):
    key_proj_weight = state_dict.pop('params/img/MAPHead_0/MultiHeadDotProductAttention_0/key/kernel').reshape(-1, config.vision_config.hidden_size).T
    key_proj_bias = state_dict.pop('params/img/MAPHead_0/MultiHeadDotProductAttention_0/key/bias').reshape(-1)
    value_proj_weight = state_dict.pop('params/img/MAPHead_0/MultiHeadDotProductAttention_0/value/kernel').reshape(-1, config.vision_config.hidden_size).T
    value_proj_bias = state_dict.pop('params/img/MAPHead_0/MultiHeadDotProductAttention_0/value/bias').reshape(-1)
    query_proj_weight = state_dict.pop('params/img/MAPHead_0/MultiHeadDotProductAttention_0/query/kernel').reshape(-1, config.vision_config.hidden_size).T
    query_proj_bias = state_dict.pop('params/img/MAPHead_0/MultiHeadDotProductAttention_0/query/bias').reshape(-1)
    state_dict['vision_model.head.attention.in_proj_weight'] = torch.from_numpy(np.concatenate([query_proj_weight, key_proj_weight, value_proj_weight], axis=0))
    state_dict['vision_model.head.attention.in_proj_bias'] = torch.from_numpy(np.concatenate([query_proj_bias, key_proj_bias, value_proj_bias], axis=0))