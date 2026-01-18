import argparse
import collections
import numpy as np
import torch
from flax import traverse_util
from t5x import checkpoints
from transformers import MT5Config, UMT5EncoderModel, UMT5ForConditionalGeneration
from transformers.utils import logging
def t5x_layer_norm_lookup(params, i, prefix, layer_name):
    """Returns the layer norm param of a layer."""
    return params[f'{prefix}/{prefix}/{layer_name}/scale'][:, i]