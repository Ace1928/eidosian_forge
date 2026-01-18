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
@register_inference_rule(torch.reshape)
def reshape_inference_rule(n: Node, symbols, constraints, counter):
    assert isinstance(n.args[0], Node)
    my_reshape, counter = gen_tvar(counter)
    symbols[n] = my_reshape
    src_var = symbols[n.args[0]]
    t2 = n.args[1]
    t2_type = TensorType([Dyn if elem == -1 else elem for elem in t2])
    c1 = BinConstraintT(my_reshape, t2_type, op_eq)
    c2 = CanReshape(src_var, t2_type)
    return ([c1, c2], counter)