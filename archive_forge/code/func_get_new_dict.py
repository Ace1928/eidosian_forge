import argparse
from collections import OrderedDict
from pathlib import Path
import torch
from transformers import (
from transformers.utils import logging
def get_new_dict(d, config, rename_keys_prefix=rename_keys_prefix):
    new_d = OrderedDict()
    new_d['visual_bert.embeddings.position_ids'] = torch.arange(config.max_position_embeddings).expand((1, -1))
    for key in d:
        if 'detector' in key:
            continue
        new_key = key
        for name_pair in rename_keys_prefix:
            new_key = new_key.replace(name_pair[0], name_pair[1])
        new_d[new_key] = d[key]
        if key == 'bert.cls.predictions.decoder.weight':
            new_d['cls.predictions.decoder.bias'] = new_d['cls.predictions.bias']
    return new_d