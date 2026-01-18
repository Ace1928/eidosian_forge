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
@register_inference_rule('index_select')
def index_select_inference_rule(n: Node, symbols, constraints, counter):
    """
    We constrain the second argument to a vector or Dyn.
    The output replaces the input with the shape of the vector
    at the position given by the index (first argument)
    """
    assert isinstance(n.args[0], Node)
    assert isinstance(n.args[1], int)
    assert isinstance(n.args[2], Node)
    index_select, counter = gen_tvar(counter)
    symbols[n] = index_select
    dims, counter = gen_tensor_dims(1, counter)
    is_size_1 = BinConstraintT(symbols[n.args[2]], TensorType(dims), op_eq)
    is_dyn = BinConstraintT(symbols[n.args[2]], Dyn, op_eq)
    c2 = Conj([is_size_1, Disj([IndexSelect(i + 1, symbols[n.args[0]], dims[0], n.args[1], index_select) for i in range(MAX_TENSOR_RANK)])])
    c3 = Conj([is_dyn, Disj([IndexSelect(i + 1, symbols[n.args[0]], Dyn, n.args[1], index_select) for i in range(MAX_TENSOR_RANK)])])
    return ([Disj([c2, c3])], counter)