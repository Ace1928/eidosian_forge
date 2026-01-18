import torch
import torch.nn as nn
from torch import Tensor
from .utils import _quantize_and_dequantize_weight
from .utils import _quantize_weight
from typing import Optional, Dict, Any, Tuple
from torch import _VF
from torch.nn.utils.rnn import PackedSequence
def get_quantized_weight(module, wn):
    if not hasattr(module, wn):
        return None
    params = _get_weight_and_quantization_params(module, wn)
    weight = _quantize_weight(*params)
    return weight