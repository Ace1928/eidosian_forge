import copy
import functools
import math
import operator
from typing import Any, Tuple
import torch
from torch._dynamo.utils import counters
from torch.fx.experimental.symbolic_shapes import has_free_symbols
from ..lowering import lowerings as L, require_channels_last
from ..pattern_matcher import Arg, CallFunction, filter_nodes, KeywordArg, ListOf, Match
from ..utils import pad_listlike
from .freezing_patterns import register_freezing_graph_pattern
from .post_grad import register_lowering_pattern
@register_lowering_pattern(pattern, extra_check=_is_valid_quantized_maxpool2d_optimization_pattern())
def qmaxpool2d(match: Match, *args, **kwargs):
    x = kwargs['x']
    kernel_size = kwargs['kernel_size']
    stride = kwargs['stride'] if 'stride' in kwargs else None
    padding = kwargs['padding'] if 'padding' in kwargs else 0
    dilation = kwargs['dilation'] if 'dilation' in kwargs else 1
    ceil_mode = kwargs['ceil_mode'] if 'ceil_mode' in kwargs else False
    if padding == 0:
        padding = [0, 0]
    if dilation == 1:
        dilation = [1, 1]
    if not stride:
        stride = kernel_size
    kernel_size = pad_listlike(kernel_size, 2)
    stride = pad_listlike(stride, 2)
    padding = pad_listlike(padding, 2)
    dilation = pad_listlike(dilation, 2)
    assert len(kernel_size) == 2
    assert len(stride) == 2
    assert len(padding) == 2
    assert len(dilation) == 2
    computation_args = (x, kernel_size, stride, padding, dilation, ceil_mode)
    computation_args, _ = require_channels_last(computation_op, *computation_args)
    return L[computation_op](*computation_args)