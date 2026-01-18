import argparse
import torch
from transformers import (
from transformers.tokenization_utils import AddedToken
def load_conv_layer(full_name, value, feature_extractor, unused_weights, use_group_norm):
    name = full_name.split('conv_layers.')[-1]
    items = name.split('.')
    layer_id = int(items[0])
    type_id = int(items[1])
    if type_id == 0:
        if 'bias' in name:
            if value.shape != feature_extractor.conv_layers[layer_id].conv.bias.data.shape:
                raise ValueError(f'{full_name} has size {value.shape}, but {feature_extractor.conv_layers[layer_id].conv.bias.data.shape} was found.')
            feature_extractor.conv_layers[layer_id].conv.bias.data = value
            logger.info(f'Feat extract conv layer {layer_id} was initialized from {full_name}.')
        elif 'weight' in name:
            if value.shape != feature_extractor.conv_layers[layer_id].conv.weight.data.shape:
                raise ValueError(f'{full_name} has size {value.shape}, but {feature_extractor.conv_layers[layer_id].conv.weight.data.shape} was found.')
            feature_extractor.conv_layers[layer_id].conv.weight.data = value
            logger.info(f'Feat extract conv layer {layer_id} was initialized from {full_name}.')
    elif type_id == 2 and (not use_group_norm) or (type_id == 2 and layer_id == 0 and use_group_norm):
        if 'bias' in name:
            if value.shape != feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape:
                raise ValueError(f'{full_name} has size {value.shape}, but {feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape} was found.')
            feature_extractor.conv_layers[layer_id].layer_norm.bias.data = value
            logger.info(f'Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.')
        elif 'weight' in name:
            if value.shape != feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape:
                raise ValueError(f'{full_name} has size {value.shape}, but {feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape} was found.')
            feature_extractor.conv_layers[layer_id].layer_norm.weight.data = value
            logger.info(f'Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.')
    else:
        unused_weights.append(full_name)