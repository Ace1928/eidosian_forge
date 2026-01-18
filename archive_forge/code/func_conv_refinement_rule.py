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
@register_refinement_rule(Conv2d)
def conv_refinement_rule(n: Node):
    """
    The equality constraints are between the first dimension of
    the input and output
    """
    res = []
    assert isinstance(n.args[0], Node)
    arg_type = n.args[0].type
    if isinstance(arg_type, TensorType) and isinstance(n.type, TensorType):
        res = [Equality(arg_type.__args__[0], n.type.__args__[0])]
        return res