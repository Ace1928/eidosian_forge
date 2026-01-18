import itertools
import logging
import operator
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union
from typing_extensions import TypeAlias
import torch
from torch._dynamo.utils import counters
from ..pattern_matcher import (
from .pre_grad import (
@register_graph_pattern(CallMethodVarArgs('squeeze', users=MULTIPLE), pass_dict=normalization_pass, extra_check=config_flag('split_cat_fx_passes'))
def normalize_squeeze_default(match: Match, *args, **kwargs):
    squeeze_node = match.nodes[0]
    squeeze_input = get_arg_value(squeeze_node, 0)
    if 'dim' in squeeze_node.kwargs:
        assert len(squeeze_node.args) == 1
        dim = squeeze_node.kwargs['dim']
    elif len(squeeze_node.args) == 1:
        dim = None
    elif len(squeeze_node.args) == 2:
        dim = squeeze_node.args[1]
    else:
        dim = squeeze_node.args[1:]
    if isinstance(dim, Sequence) and len(dim) == 1:
        dim = dim[0]
    with match.graph.inserting_after(squeeze_node):
        if dim is None:
            new_squeeze_node = match.graph.call_function(torch.squeeze, args=(squeeze_input,))
        else:
            new_squeeze_node = match.graph.call_function(torch.squeeze, args=(squeeze_input,), kwargs={'dim': dim})
    squeeze_node.replace_all_uses_with(new_squeeze_node)
    match.graph.erase_node(squeeze_node)