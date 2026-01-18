import torch
import operator
import warnings
from typing import Callable, Dict, Iterable
from torch.fx._symbolic_trace import _assert_is_none
from torch.fx.experimental.migrate_gradual_types.constraint import ApplyBroadcasting, CalcProduct, \
from torch.fx.experimental.migrate_gradual_types.operation import \
from torch.fx.node import Target, Node
from torch.fx.experimental.migrate_gradual_types.util import gen_tensor_dims, gen_nat_constraints, gen_dvar, gen_tvar, \
from torch.fx.tensor_type import Dyn, TensorType
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
@register_inference_rule(torch.full)
def full_inference_rule(n: Node, symbols, constraints, counter):
    full, counter = gen_tvar(counter)
    symbols[n] = full
    res = []
    assert isinstance(n.args[0], Iterable)
    for arg in n.args[0]:
        dim = arg if isinstance(arg, int) else symbols[arg]
        res.append(dim)
    c = BinConstraintT(full, TensorType(list(res)), op_eq)
    return ([c], counter)