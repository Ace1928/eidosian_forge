import operator
from functools import partial
from typing import Any, Callable, Dict
from sympy import Expr
import torch
from torch.utils._sympy.value_ranges import bound_sympy, ValueRangeAnalysis, ValueRanges
from .ir import InterpreterShim, LoopBody, LoopBodyBlock
from .utils import cache_on_self, dominated_nodes
from .virtualized import V
def masked_subblock(self, subblock: LoopBodyBlock, env: Dict[torch.fx.Node, ValueRanges], mask: Any, value: Any, submodules: Dict[str, Callable[..., Any]]) -> ValueRanges:
    interp = InterpreterShim(subblock.graph, submodules)
    interp.run(V.get_ops_handler(), initial_env=env)
    output = [node for node in subblock.graph.nodes if node.target == 'output']
    assert len(output) == 1
    return interp.env[output[0]]