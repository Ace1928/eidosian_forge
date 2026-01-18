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
def clone_to_new_node(graph, source_node, user_node):
    assert source_node.op == 'call_function', 'clone_to_new_node only support node.op call_function'
    with graph.inserting_before(user_node):
        new_node = graph.call_function(source_node.target, args=source_node.args, kwargs=source_node.kwargs)
        new_node.meta = copy.copy(source_node.meta)
        user_node.replace_input_with(source_node, new_node)
    return new_node