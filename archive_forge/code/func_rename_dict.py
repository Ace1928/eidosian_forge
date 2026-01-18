import argparse
import json
import os
import fairseq
import torch
from fairseq.data import Dictionary
from transformers import (
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2ForSequenceClassification
def rename_dict(key, value, full_name, weight_type, hf_dict):
    hf_param_name = None
    for param_key in PARAM_MAPPING.keys():
        if full_name.endswith(param_key):
            hf_param_name = PARAM_MAPPING[full_name.split('.')[-1]]
            weight_type = 'param'
    if weight_type is not None and weight_type != 'param':
        full_key = '.'.join([key, weight_type])
    elif weight_type is not None and weight_type == 'param':
        full_key = '.'.join([key, hf_param_name])
    else:
        full_key = key
    hf_dict[full_key] = value if 'lm_head' in full_key else value[0]