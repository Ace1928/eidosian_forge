import itertools
import logging
import operator
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union
from typing_extensions import TypeAlias
import torch
from torch._dynamo.utils import counters
from ..pattern_matcher import (
from .pre_grad import (
@register_graph_pattern(CallFunctionVarArgs(torch.cat, users=MULTIPLE), pass_dict=normalization_pass, extra_check=config_flag('split_cat_fx_passes'))
def normalize_cat_default(match: Match, *args, **kwargs):
    cat_node = match.nodes[0]
    graph = match.graph
    tensors = get_arg_value(cat_node, 0, 'tensors')
    cat_dim = get_arg_value(cat_node, 1, 'dim')
    if cat_dim is None:
        cat_axis = cat_node.kwargs.get('axis')
        if cat_axis is not None:
            cat_dim = cat_axis
        else:
            cat_dim = 0
    if tensors is None or cat_dim is None:
        log.info("couldn't find cat args")
        return
    assert isinstance(tensors, (list, tuple))
    for tensor in itertools.chain([cat_node], tensors):
        if 'example_value' not in tensor.meta:
            log.warning('example value absent for node: %s', tensor)
            return
    ndim = cat_node.meta['example_value'].dim()

    def is_empty_tensor(x):
        x_shape = x.meta['example_value'].shape
        return len(x_shape) == 1 and x_shape[0] == 0
    assert all((ndim == x.meta['example_value'].dim() or is_empty_tensor(x) for x in tensors))
    if cat_dim < 0:
        cat_dim += ndim
    with graph.inserting_after(cat_node):
        new_cat_node = graph.call_function(torch.cat, args=(tensors,), kwargs={'dim': cat_dim})
    cat_node.replace_all_uses_with(new_cat_node)
    new_cat_node.meta.update(cat_node.meta)
    graph.erase_node(cat_node)
    counters['inductor']['split_cat_norm'] += 1