import collections
import logging
import torch
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from .. import config, inductor_prims
from ..pattern_matcher import (
from ..virtualized import V
@register_graph_pattern(CallFunctionVarArgs(aten.randint.low), pass_dict=patterns)
def replace_randint(match: Match, low, high, size, *, dtype=torch.int64, device=None, layout=None, pin_memory=None):

    def replacement(size):
        result = inductor_prims.randint(low, high, size, inductor_prims.seed(device))
        return result.to(dtype)
    device = get_device(device)
    match.replace_by_example(replacement, [size])