import functools
import itertools
import torch
from ..._dynamo.utils import counters
from ..pattern_matcher import Arg, CallFunction, KeywordArg
from .freezing_patterns import register_binary_folding_pattern
def mark_mixed_dtype_allowed_convs(gm):
    """
    Mark convolutions which we will binary fold even with mixed precision constants. We constant fold in the higher precision
    for better accuracy and then recover the original precision after.
    """
    for node in gm.graph.nodes:
        if node.target is aten.convolution.default:
            mark_mixed_dtype_conv(node)