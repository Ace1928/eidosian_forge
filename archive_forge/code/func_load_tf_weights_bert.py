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
def load_tf_weights_bert(init_vars, tf_path):
    names = []
    tf_weights = {}
    for name, shape in init_vars:
        array = tf.train.load_variable(tf_path, name)
        name = name.replace('bert/encoder/LayerNorm', 'bert/embeddings/LayerNorm')
        logger.info(f'Loading TF weight {name} with shape {shape}')
        names.append(name)
        tf_weights[name] = array
    return (names, tf_weights)