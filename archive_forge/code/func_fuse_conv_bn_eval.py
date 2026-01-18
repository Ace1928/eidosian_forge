from __future__ import annotations
import copy
from typing import Optional, Tuple, TypeVar
import torch
def fuse_conv_bn_eval(conv: ConvT, bn: torch.nn.modules.batchnorm._BatchNorm, transpose: bool=False) -> ConvT:
    """Fuse a convolutional module and a BatchNorm module into a single, new convolutional module.

    Args:
        conv (torch.nn.modules.conv._ConvNd): A convolutional module.
        bn (torch.nn.modules.batchnorm._BatchNorm): A BatchNorm module.
        transpose (bool, optional): If True, transpose the convolutional weight. Defaults to False.

    Returns:
        torch.nn.modules.conv._ConvNd: The fused convolutional module.

    .. note::
        Both ``conv`` and ``bn`` must be in eval mode, and ``bn`` must have its running buffers computed.
    """
    assert not (conv.training or bn.training), 'Fusion only for eval!'
    fused_conv = copy.deepcopy(conv)
    assert bn.running_mean is not None and bn.running_var is not None
    fused_conv.weight, fused_conv.bias = fuse_conv_bn_weights(fused_conv.weight, fused_conv.bias, bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias, transpose)
    return fused_conv