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
@register_lowering_pattern(pattern, extra_check=_is_input_output_same_scale_zp(aten.cat.default))
def qcat(match: Match, inputs, dim, **kwargs):
    uint8_inputs = [input[0] for input in inputs]
    return L[computation_op](uint8_inputs, dim)