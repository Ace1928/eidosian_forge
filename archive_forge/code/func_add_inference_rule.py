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
@register_inference_rule(torch.add)
@register_inference_rule(operator.add)
def add_inference_rule(n: Node):
    """
    Apply the addition inference rule. This includes:
    - scalar addition
    - broadcasting semantics

    Note that we always return the least precise type between
    the operands (after applying broadcasting) to be the final type of the operation

    Note that we do not modify the operand types themselves after applying broadcasting
    to them. We only use them to calculate the final type
    """
    assert isinstance(n.args[0], Node)
    assert isinstance(n.args[1], Node)
    t1 = n.args[0].type
    t2 = n.args[1].type
    if t1 == int and isinstance(t2, TensorType):
        n.type = t2
        return n.type
    elif t2 == int and isinstance(t1, TensorType):
        n.type = t1
        return n.type
    new_t1, new_t2 = broadcast_types(t1, t2)
    if new_t1 != t1 or new_t2 != t2:
        n.meta['broadcast'] = True
        n.meta[str(n.args[0])] = new_t1
        n.meta[str(n.args[1])] = new_t2
    else:
        n.meta['broadcast'] = False
    new_t1 = t1 if not n.meta['broadcast'] else new_t1
    new_t2 = t2 if not n.meta['broadcast'] else new_t2
    if is_consistent(new_t1, new_t2):
        if is_more_precise(new_t1, new_t2):
            n.type = new_t2
        else:
            n.type = new_t1
        return n.type
    else:
        raise TypeError(f'Cannot add arguments {n.args[0]} ({n.args[0].type}) and {n.args[1]} ({n.args[1].type}) in node {n}. Types should match ')