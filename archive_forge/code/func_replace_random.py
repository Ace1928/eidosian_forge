import collections
import logging
import torch
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from .. import config, inductor_prims
from ..pattern_matcher import (
from ..virtualized import V
@register_graph_pattern(CallFunctionVarArgs(aten.rand.default), pass_dict=patterns)
@register_graph_pattern(CallFunctionVarArgs(aten.rand.generator), pass_dict=patterns)
@register_graph_pattern(CallFunctionVarArgs(aten.randn.default), pass_dict=patterns)
@register_graph_pattern(CallFunctionVarArgs(aten.randn.generator), pass_dict=patterns)
def replace_random(match: Match, size, *, generator=None, dtype=None, device=None, layout=None, pin_memory=None):
    if generator is not None:
        return

    def replacement(size):
        result = inductor_prims.random(size, inductor_prims.seed(device), mode, **default_kwargs(device))
        if dtype is not None:
            result = result.to(dtype)
        return result
    mode = {aten.rand: 'rand', aten.randn: 'randn'}[match.output_node().target.overloadpacket]
    device = get_device(device)
    match.replace_by_example(replacement, [size])