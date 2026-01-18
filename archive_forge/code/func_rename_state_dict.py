import argparse
import torch
from transformers import UnivNetConfig, UnivNetModel, logging
def rename_state_dict(state_dict, keys_to_modify, keys_to_remove):
    model_state_dict = {}
    for key, value in state_dict.items():
        if key in keys_to_remove:
            continue
        if key in keys_to_modify:
            new_key = keys_to_modify[key]
            model_state_dict[new_key] = value
        else:
            model_state_dict[key] = value
    return model_state_dict