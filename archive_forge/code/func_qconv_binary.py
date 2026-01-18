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
@register_lowering_pattern(pattern, extra_check=_is_valid_quantized_conv_binary_optimization_pattern(output_dtype), pass_number=pass_number)
def qconv_binary(match: Match, *args, **kwargs):
    x, x_scale, x_zp = (kwargs['x'], kwargs['x_scale'], kwargs['x_zp'])
    accum = kwargs['accum'] if output_dtype is None else kwargs['accum_after_dequant']
    accum_scale = kwargs['accum_scale'] if output_dtype is None else 1.0
    accum_zp = kwargs['accum_zp'] if output_dtype is None else 0
    packed_weight, w_scale, w_zp = (kwargs['packed_weight'], kwargs['w_scale'], kwargs['w_zp'])
    b, stride, padding, dilation, groups = (kwargs['b'], kwargs['stride'], kwargs['padding'], kwargs['dilation'], kwargs['groups'])
    o_inv_scale = kwargs['o_inv_scale'] if output_dtype is None else 1.0
    o_zero_point = kwargs['o_zp'] if output_dtype is None else 0
    computation_args = (x, x_scale, x_zp, accum, accum_scale, accum_zp, packed_weight, w_scale, w_zp, b, stride, padding, dilation, groups, o_inv_scale, o_zero_point, output_dtype, binary_unary_attr.binary_op_name, binary_unary_attr.alpha, binary_unary_attr.unary_op_name, binary_unary_attr.scalars_attr, binary_unary_attr.algorithm_attr)
    counters['inductor']['qconv2d_binary_matcher_count'] += 1
    counters['inductor']['qconv2d_binary_matcher_nodes'] += len(match.nodes)
    return L[computation_op](*computation_args)