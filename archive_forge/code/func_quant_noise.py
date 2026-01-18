import math
from typing import Iterable, Iterator, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_graphormer import GraphormerConfig
def quant_noise(module: nn.Module, p: float, block_size: int):
    """
    From:
    https://github.com/facebookresearch/fairseq/blob/dd0079bde7f678b0cd0715cbd0ae68d661b7226d/fairseq/modules/quant_noise.py

    Wraps modules and applies quantization noise to the weights for subsequent quantization with Iterative Product
    Quantization as described in "Training with Quantization Noise for Extreme Model Compression"

    Args:
        - module: nn.Module
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ

    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights, see "And the Bit Goes Down:
          Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper which consists in randomly dropping
          blocks
    """
    if p <= 0:
        return module
    if not isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d)):
        raise NotImplementedError('Module unsupported for quant_noise.')
    is_conv = module.weight.ndim == 4
    if not is_conv:
        if module.weight.size(1) % block_size != 0:
            raise AssertionError('Input features must be a multiple of block sizes')
    elif module.kernel_size == (1, 1):
        if module.in_channels % block_size != 0:
            raise AssertionError('Input channels must be a multiple of block sizes')
    else:
        k = module.kernel_size[0] * module.kernel_size[1]
        if k % block_size != 0:
            raise AssertionError('Kernel size must be a multiple of block size')

    def _forward_pre_hook(mod, input):
        if mod.training:
            if not is_conv:
                weight = mod.weight
                in_features = weight.size(1)
                out_features = weight.size(0)
                mask = torch.zeros(in_features // block_size * out_features, device=weight.device)
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)
            else:
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels
                if mod.kernel_size == (1, 1):
                    mask = torch.zeros(int(in_channels // block_size * out_channels), device=weight.device)
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
                else:
                    mask = torch.zeros(weight.size(0), weight.size(1), device=weight.device)
                    mask.bernoulli_(p)
                    mask = mask.unsqueeze(2).unsqueeze(3).repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])
            mask = mask.to(torch.bool)
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0)
    module.register_forward_pre_hook(_forward_pre_hook)
    return module