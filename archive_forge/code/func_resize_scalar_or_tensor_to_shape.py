import functools
import itertools
import torch
from ..._dynamo.utils import counters
from ..pattern_matcher import Arg, CallFunction, KeywordArg
from .freezing_patterns import register_binary_folding_pattern
def resize_scalar_or_tensor_to_shape(graph, other, shape):
    if other.meta.get('val').numel() == 1:
        res = graph.create_node('call_function', aten.reshape.default, (other, (1,)))
        res = graph.create_node('call_function', aten.expand.default, (res, shape))
    else:
        res = graph.create_node('call_function', aten.reshape.default, (other, shape))
    return res