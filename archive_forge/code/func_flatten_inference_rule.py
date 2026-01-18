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
@register_inference_rule(torch.flatten)
def flatten_inference_rule(n: Node, symbols, constraints, counter):
    assert isinstance(n.args[0], Node)
    flattened, counter = gen_tvar(counter)
    symbols[n] = flattened
    input = symbols[n.args[0]]
    start_dim = 1
    end_dim = -1
    if len(n.args) > 1:
        assert isinstance(n.args[1], int)
        start_dim = n.args[1]
    if len(n.args) > 2:
        assert isinstance(n.args[2], int)
        end_dim = n.args[2]
    c1 = BinConstraintT(input, Dyn, op_eq)
    c2 = BinConstraintT(flattened, Dyn, op_eq)
    both_dyn = Conj([c1, c2])
    const = []
    for i in range(1, MAX_TENSOR_RANK + 1):
        c, counter = generate_flatten_constraints(start_dim, end_dim, input, flattened, i, counter)
        const.append(c)
    return ([Disj([both_dyn, *const])], counter)