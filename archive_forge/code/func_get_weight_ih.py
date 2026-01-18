import torch
import torch.nn as nn
from torch import Tensor
from .utils import _quantize_and_dequantize_weight
from .utils import _quantize_weight
from typing import Optional, Dict, Any, Tuple
from torch import _VF
from torch.nn.utils.rnn import PackedSequence
def get_weight_ih(self):
    return _get_quantize_and_dequantized_weight(self, 'weight_ih')