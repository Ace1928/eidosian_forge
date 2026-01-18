import argparse
import re
import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import (
def replace_keys(state_dict):
    model_state_dict = {}
    state_dict.pop('pixel_mean', None)
    state_dict.pop('pixel_std', None)
    output_hypernetworks_mlps_pattern = '.*.output_hypernetworks_mlps.(\\d+).layers.(\\d+).*'
    for key, value in state_dict.items():
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)
        if re.match(output_hypernetworks_mlps_pattern, key):
            layer_nb = int(re.match(output_hypernetworks_mlps_pattern, key).group(2))
            if layer_nb == 0:
                key = key.replace('layers.0', 'proj_in')
            elif layer_nb == 1:
                key = key.replace('layers.1', 'layers.0')
            elif layer_nb == 2:
                key = key.replace('layers.2', 'proj_out')
        model_state_dict[key] = value
    model_state_dict['shared_image_embedding.positional_embedding'] = model_state_dict['prompt_encoder.shared_embedding.positional_embedding']
    return model_state_dict