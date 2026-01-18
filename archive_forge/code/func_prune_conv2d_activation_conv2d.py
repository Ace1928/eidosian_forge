from typing import cast, List, Optional, Callable, Tuple
import torch
from torch import nn, Tensor
from torch.nn.utils import parametrize
from torch.nn.utils.parametrize import ParametrizationList
from .parametrization import FakeStructuredSparsity, BiasHook
def prune_conv2d_activation_conv2d(conv2d_1: nn.Conv2d, activation: Optional[Callable[[Tensor], Tensor]], conv2d_2: nn.Conv2d):
    """
    Fusion Pattern for conv2d -> some activation module / function -> conv2d layers
    """
    parametrization_dict = cast(nn.ModuleDict, conv2d_1.parametrizations)
    weight_parameterizations = cast(ParametrizationList, parametrization_dict.weight)
    for p in weight_parameterizations:
        if isinstance(p, FakeStructuredSparsity):
            mask = cast(Tensor, p.mask)
    prune_bias = getattr(conv2d_1, 'prune_bias', False)
    if hasattr(conv2d_2, 'padding') and cast(Tuple[int], conv2d_2.padding) > (0, 0) and (conv2d_1.bias is not None or getattr(conv2d_1, '_bias', None) is not None):
        prune_conv2d_padded(conv2d_1)
    else:
        mask = _prune_conv2d_helper(conv2d_1)
        if prune_bias:
            _prune_module_bias(conv2d_1, mask)
        else:
            pruned_biases = _propogate_module_bias(conv2d_1, mask)
            if pruned_biases is not None:
                if activation:
                    pruned_biases = activation(pruned_biases)
                conv2d_2.bias = _get_adjusted_next_layer_bias(conv2d_2, pruned_biases, mask)
        if not (hasattr(conv2d_2, 'padding') and cast(Tuple[int], conv2d_2.padding) > (0, 0)) or conv2d_1.bias is None:
            with torch.no_grad():
                if parametrize.is_parametrized(conv2d_2):
                    parametrization_dict = cast(nn.ModuleDict, conv2d_2.parametrizations)
                    weight_parameterizations = cast(ParametrizationList, parametrization_dict.weight)
                    weight_parameterizations.original = nn.Parameter(weight_parameterizations.original[:, mask])
                    conv2d_2.in_channels = weight_parameterizations.original.shape[1]
                else:
                    conv2d_2.weight = nn.Parameter(conv2d_2.weight[:, mask])
                    conv2d_2.in_channels = conv2d_2.weight.shape[1]