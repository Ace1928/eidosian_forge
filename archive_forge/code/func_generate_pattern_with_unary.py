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
def generate_pattern_with_unary(computation_call, unary_post_op):
    if unary_post_op is not None:
        if unary_post_op == aten.hardtanh.default:
            return CallFunction(aten.clamp_max, CallFunction(aten.clamp_min, computation_call, KeywordArg('min_value')), KeywordArg('max_value'))
        else:
            return CallFunction(unary_post_op, computation_call)
    return computation_call