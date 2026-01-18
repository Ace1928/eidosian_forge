import argparse
import json
import os
import fairseq
import torch
from torch import nn
from transformers import (
def recursively_load_weights_wav2vec2(fairseq_model, hf_model):
    unused_weights = []
    fairseq_dict = fairseq_model.state_dict()
    feature_extractor = hf_model.feature_extractor
    proj_weight = None
    for name, value in fairseq_dict.items():
        is_used = False
        if 'conv_layers' in name:
            load_conv_layer(name, value, feature_extractor, unused_weights, hf_model.config.feat_extract_norm == 'group')
            is_used = True
        elif name.split('.')[0] == 'proj':
            proj_weight = fairseq_model.proj
            is_used = True
        else:
            for key, mapped_key in MAPPING.items():
                if key in name or key.split('w2v_model.')[-1] == name.split('.')[0]:
                    is_used = True
                    if '*' in mapped_key:
                        layer_index = name.split(key)[0].split('.')[-2]
                        mapped_key = mapped_key.replace('*', layer_index)
                    if 'weight_g' in name:
                        weight_type = 'weight_g'
                    elif 'weight_v' in name:
                        weight_type = 'weight_v'
                    elif 'bias' in name:
                        weight_type = 'bias'
                    elif 'weight' in name:
                        weight_type = 'weight'
                    else:
                        weight_type = None
                    set_recursively(hf_model, mapped_key, value, name, weight_type)
                continue
        if not is_used:
            unused_weights.append(name)
    logger.warning(f'Unused weights: {unused_weights}')
    return proj_weight