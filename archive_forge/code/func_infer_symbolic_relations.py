from functools import reduce
import torch
import operator
from torch.fx.tensor_type import Dyn, is_consistent, TensorType, is_more_precise
from typing import Callable, Dict
from torch.fx.node import Target, Node
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d
from torch.fx.experimental.refinement_types import Equality
import itertools
from torch.fx.experimental.unification import Var  # type: ignore[attr-defined]
import sympy
def infer_symbolic_relations(self, n: Node):
    n.type = self.convert_to_sympy_symbols(n.type)
    if n.op == 'call_function':
        if n.target in _RULES:
            return _RULES[n.target](n)
        else:
            pass
    if n.op == 'call_module':
        module_instance = self.traced.get_submodule(n.target)
        if type(module_instance) in _RULES:
            return _RULES[type(module_instance)](n, module_instance)
        else:
            pass
    if n.op == 'output':

        def get_node_type(a):
            return a.type
        n.type = torch.fx.node.map_arg(n.args[0], get_node_type)
        return n.type
    else:
        pass