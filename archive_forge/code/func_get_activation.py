import math
import warnings
from collections import OrderedDict
import torch
from packaging import version
from torch import Tensor, nn
from .utils import logging
def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(f'function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}')