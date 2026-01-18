import torch
import torch.nn as nn
from torch import Tensor
from .utils import _quantize_and_dequantize_weight
from .utils import _quantize_weight
from typing import Optional, Dict, Any, Tuple
from torch import _VF
from torch.nn.utils.rnn import PackedSequence
def get_flat_weights(self):
    flat_weights = []
    for wn in self._flat_weights_names:
        if hasattr(self, wn):
            weight = getattr(self, wn)
            if wn.startswith('weight'):
                params = _get_weight_and_quantization_params(self, wn)
                weight = _quantize_and_dequantize_weight(*params)
        else:
            weight = None
        flat_weights.append(weight)
    return flat_weights