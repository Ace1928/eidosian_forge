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
@register_inference_rule(getattr)
def get_attr_inference_rule(n: Node, symbols, constraints, counter):
    """
    If the attribute is "device" then the tensor shape is preserved
    """
    assert isinstance(n.args[0], Node)
    assert isinstance(n.args[1], str)
    output, counter = gen_tvar(counter)
    symbols[n] = output
    input = symbols[n.args[0]]
    attr = n.args[1]
    if attr == 'device':
        return ([BinConstraintT(input, output, op_eq)], counter)
    else:
        raise NotImplementedError('Not yet implemented')