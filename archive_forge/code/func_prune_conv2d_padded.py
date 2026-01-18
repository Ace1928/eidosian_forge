from typing import cast, List, Optional, Callable, Tuple
import torch
from torch import nn, Tensor
from torch.nn.utils import parametrize
from torch.nn.utils.parametrize import ParametrizationList
from .parametrization import FakeStructuredSparsity, BiasHook
def prune_conv2d_padded(conv2d_1: nn.Conv2d) -> None:
    parametrization_dict = cast(nn.ModuleDict, conv2d_1.parametrizations)
    weight_parameterizations = cast(ParametrizationList, parametrization_dict.weight)
    for p in weight_parameterizations:
        if isinstance(p, FakeStructuredSparsity):
            mask = cast(Tensor, p.mask)
    with torch.no_grad():
        parametrize.remove_parametrizations(conv2d_1, 'weight', leave_parametrized=True)
    if getattr(conv2d_1, '_bias', None) is not None:
        if conv2d_1.bias is not None:
            new_bias = torch.zeros(conv2d_1.bias.shape)
            new_bias[mask] = conv2d_1.bias[mask]
            new_bias[~mask] = cast(Tensor, conv2d_1._bias)[~mask]
            conv2d_1.bias = nn.Parameter(new_bias)
        else:
            conv2d_1.bias = nn.Parameter(cast(Tensor, conv2d_1._bias))
    elif conv2d_1.bias is not None:
        conv2d_1.bias.data[~mask] = 0
    if hasattr(conv2d_1, '_bias'):
        delattr(conv2d_1, '_bias')