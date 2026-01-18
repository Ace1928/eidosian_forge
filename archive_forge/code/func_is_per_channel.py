import functools
import warnings
from collections import OrderedDict
from inspect import getfullargspec, signature
from typing import Any, Callable, Dict, Optional, Tuple, Union
import torch
from torch.ao.quantization.quant_type import QuantType
from torch.fx import Node
from torch.nn.utils.parametrize import is_parametrized
def is_per_channel(qscheme):
    return qscheme in [torch.per_channel_affine, torch.per_channel_affine_float_qparams, torch.per_channel_symmetric]