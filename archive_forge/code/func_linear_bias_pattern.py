import functools
import operator
from functools import reduce
from typing import Any, Tuple
import torch
from torch.fx.experimental.symbolic_shapes import has_free_symbols
from .. import ir
from ..lowering import lowerings as L
from ..pattern_matcher import (
from ..virtualized import ops
from .freezing_patterns import register_freezing_graph_pattern
from .post_grad import register_lowering_pattern
from .quantization import (
@register_freezing_graph_pattern(CallFunction(aten.add.Tensor, CallFunction(mkldnn._linear_pointwise.default, *_linear_args), Arg()), pass_number=1, extra_check=is_linear_add_bias)
def linear_bias_pattern(match, *args):
    graph = match.graph
    add_node = match.output_node()
    linear_node = add_node.args[0]
    new_args = list(linear_node.args)
    new_args[2] = add_node.args[1]
    repl = graph.call_function(mkldnn._linear_pointwise.default, tuple(new_args))
    repl.meta.update(add_node.meta)
    add_node.replace_all_uses_with(repl)
    match.erase_nodes(graph)