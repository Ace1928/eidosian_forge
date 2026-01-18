import logging
from typing import List, Optional, Tuple
import torch
from torch import nn, Tensor
from torch.nn import Module, Parameter
from .wavlm_attention import WavLMSelfAttention
def get_intermediate_outputs(self, x: Tensor, attention_mask: Optional[Tensor]=None, num_layers: Optional[int]=None) -> List[Tensor]:
    if num_layers is not None:
        if not 0 < num_layers <= len(self.layers):
            raise ValueError(f'`num_layers` must be between [1, {len(self.layers)}]')
    ret: List[Tensor] = []
    position_bias = None
    x = self._preprocess(x)
    for layer in self.layers:
        x, position_bias = layer(x, attention_mask, position_bias=position_bias)
        ret.append(x)
        if num_layers is not None and len(ret) >= num_layers:
            return ret
    return ret