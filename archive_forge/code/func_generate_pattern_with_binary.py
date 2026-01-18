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
def generate_pattern_with_binary(binary_post_op, computation_call, extra_input_pattern, int8_mixed_bf16_with_inplace_add=False):
    binary_pattern = CallFunction(binary_post_op, computation_call, extra_input_pattern)
    return _may_generate_pattern_with_dtype_convert(binary_pattern, KeywordArg('convert_dtype_after_inplace_add'), int8_mixed_bf16_with_inplace_add)