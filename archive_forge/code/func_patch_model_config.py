from enum import Enum
from typing import Dict, Union
import pytorch_lightning as pl
import torch
import torch.nn as nn
from xformers.components import build_attention
from xformers.components.multi_head_dispatch import MultiHeadDispatchConfig
from xformers.factory import xFormer, xFormerConfig, xFormerEncoderConfig
from xformers.utils import generate_matching_config
def patch_model_config(config, attention_name):
    commons = config['common']
    try:
        extra_attention_settings = config['extra_settings']['attention'][attention_name]
    except KeyError:
        extra_attention_settings = None
    for bc in config['xformer']:
        bc['dim_model'] = commons['dim_model']
        bc['position_encoding_config'].update(commons)
        bc['feedforward_config'].update(commons)
        bc['multi_head_config'].update(commons)
        bc['multi_head_config']['attention'].update(commons)
        bc['multi_head_config']['attention']['name'] = attention_name
        bc['multi_head_config']['attention']['dim_head'] = commons['dim_model'] / commons['num_heads']
        if extra_attention_settings is not None:
            bc['multi_head_config']['attention'].update(extra_attention_settings)
        bc['multi_head_config'] = generate_matching_config(bc['multi_head_config'], MultiHeadDispatchConfig)
        bc['multi_head_config'].attention = build_attention(bc['multi_head_config'].attention)
        bc = generate_matching_config(bc, xFormerEncoderConfig)
    return config