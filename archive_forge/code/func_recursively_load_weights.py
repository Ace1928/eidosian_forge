import argparse
import torch
from transformers import (
from transformers.tokenization_utils import AddedToken
def recursively_load_weights(fairseq_dict, hf_model, task):
    unused_weights = []
    if task == 's2t':
        feature_encoder = hf_model.speecht5.encoder.prenet.feature_encoder
        MAPPING = MAPPING_S2T
        IGNORE_KEYS = IGNORE_KEYS_S2T
    elif task == 't2s':
        feature_encoder = None
        MAPPING = MAPPING_T2S
        IGNORE_KEYS = IGNORE_KEYS_T2S
    elif task == 's2s':
        feature_encoder = hf_model.speecht5.encoder.prenet.feature_encoder
        MAPPING = MAPPING_S2S
        IGNORE_KEYS = IGNORE_KEYS_S2S
    else:
        raise ValueError(f'Unsupported task: {task}')
    for name, value in fairseq_dict.items():
        if should_ignore(name, IGNORE_KEYS):
            logger.info(f'{name} was ignored')
            continue
        is_used = False
        if 'conv_layers' in name:
            load_conv_layer(name, value, feature_encoder, unused_weights, hf_model.config.feat_extract_norm == 'group')
            is_used = True
        else:
            for key, mapped_key in MAPPING.items():
                if '*' in key:
                    prefix, suffix = key.split('.*.')
                    if prefix in name and suffix in name:
                        key = suffix
                if key in name:
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
                    elif 'running_mean' in name:
                        weight_type = 'running_mean'
                    elif 'running_var' in name:
                        weight_type = 'running_var'
                    elif 'num_batches_tracked' in name:
                        weight_type = 'num_batches_tracked'
                    else:
                        weight_type = None
                    set_recursively(hf_model, mapped_key, value, name, weight_type)
                continue
        if not is_used:
            unused_weights.append(name)
    logger.warning(f'Unused weights: {unused_weights}')