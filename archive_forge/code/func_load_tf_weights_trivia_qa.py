import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_big_bird import BigBirdConfig
def load_tf_weights_trivia_qa(init_vars):
    names = []
    tf_weights = {}
    for i, var in enumerate(init_vars):
        name_items = var.name.split('/')
        if 'transformer_scaffold' in name_items[0]:
            layer_name_items = name_items[0].split('_')
            if len(layer_name_items) < 3:
                layer_name_items += [0]
            name_items[0] = f'bert/encoder/layer_{layer_name_items[2]}'
        name = '/'.join([_TRIVIA_QA_MAPPING[x] if x in _TRIVIA_QA_MAPPING else x for x in name_items])[:-2]
        if 'self/attention/output' in name:
            name = name.replace('self/attention/output', 'output')
        if i >= len(init_vars) - 2:
            name = name.replace('intermediate', 'output')
        logger.info(f'Loading TF weight {name} with shape {var.shape}')
        array = var.value().numpy()
        names.append(name)
        tf_weights[name] = array
    return (names, tf_weights)