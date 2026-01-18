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
@register_refinement_rule(torch.add)
@register_refinement_rule(operator.add)
def element_wise_eq(n: Node):
    """
    For element-wise operations and handles broadcasting.
    Note that after applying broadcasting to the arguments
    we are able to determine if certain dimensions have not been broadcast
    if they are symbolicallu equal.

    in this case, we can establish equality between those dimensions and the
    corresponding output dimensions.

    Note that it takes two iterations for this result. One iteration to establish
    equality between certain dimensions of the operands (requiring the whole solver
    including unification) and another iteration to establish equality between the operands
    and the resulting type, requiring another round of constraint generation and unificaiton.
    """
    res = []
    if isinstance(n.args[0], Node) and isinstance(n.args[1], Node):
        arg_type1 = n.args[0].type
        arg_type2 = n.args[1].type
        if isinstance(arg_type1, TensorType) and isinstance(arg_type2, TensorType) and isinstance(n.type, TensorType):
            args1, args2 = broadcast_types(arg_type1, arg_type2)
            a1 = args1.__args__
            a2 = args2.__args__
            a3 = n.type.__args__
            r = []
            for x, y, z in zip(a1, a2, a3):
                if x == y:
                    r.append(Equality(x, z))
            res = r
    return res