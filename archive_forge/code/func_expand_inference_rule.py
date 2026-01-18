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
@register_inference_rule('expand')
def expand_inference_rule(n: Node, symbols, constraints, counter):
    """
    We generate the exact constraints as we do for tensor additions but we constraint
    the rank of this expression to be equal to len(n.args[1:]) so that only
    those cases get considered for the output
    """
    assert isinstance(n.args[0], Node)
    expand, counter = gen_tvar(counter)
    symbols[n] = expand
    e1 = symbols[n.args[0]]
    e2, counter = gen_tvar(counter)
    e2_nat_constraints = []
    for arg in n.args[1:]:
        assert isinstance(arg, (Node, int))
        if isinstance(arg, Node):
            assert isinstance(symbols[arg], DVar)
            e2_nat_constraints.append(BinConstraintD(0, symbols[arg], op_leq))
    e2_constraint = BinConstraintT(e2, TensorType([arg if isinstance(arg, int) else symbols[arg] for arg in n.args[1:]]), op_eq)
    constraints, counter = gen_broadcasting_constraints(e1, e2, symbols, counter, expand)
    dims, counter = gen_tensor_dims(len(n.args[1:]), counter)
    nat_constraints = gen_nat_constraints(dims)
    c = [BinConstraintT(expand, TensorType(dims), op_eq), *nat_constraints, e2_constraint, *e2_nat_constraints]
    constraints += c
    return (constraints, counter)