import torch
import torch.nn as nn
from torch import Tensor
from .utils import _quantize_and_dequantize_weight
from .utils import _quantize_weight
from typing import Optional, Dict, Any, Tuple
from torch import _VF
from torch.nn.utils.rnn import PackedSequence
def get_quantized_weight_bias_dict(self):
    """ dictionary from flat_weight_name to quantized weight or (unquantized) bias
        e.g.
        {
          "weight_ih_l0": quantized_weight,
          "bias_ih_l0": unquantized_bias,
          ...
        }
        """
    quantized_weight_bias_dict = {}
    for wn in self._flat_weights_names:
        if hasattr(self, wn):
            if wn.startswith('weight'):
                weight_or_bias = get_quantized_weight(self, wn)
            else:
                weight_or_bias = getattr(self, wn)
        else:
            weight_or_bias = None
        quantized_weight_bias_dict[wn] = weight_or_bias
    return quantized_weight_bias_dict