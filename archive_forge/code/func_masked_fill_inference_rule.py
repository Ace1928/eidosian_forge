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
@register_inference_rule('masked_fill_')
def masked_fill_inference_rule(n: Node, symbols, constraints, counter):
    """
    Similar to addition. For now we implement the constraints when
    the argument is a boolean tensor. There is also a case for when
    it is a condition. We will leave this out for now.
    """
    assert isinstance(n.args[0], Node)
    assert isinstance(n.args[1], Node)
    e1 = symbols[n.args[0]]
    e2 = symbols[n.args[1]]
    if isinstance(e1, TVar) and isinstance(e2, TVar):
        masked_fill_tensor, counter = gen_tvar(counter)
        symbols[n] = masked_fill_tensor
        return gen_broadcasting_constraints(e1, e2, symbols, counter, masked_fill_tensor)
    else:
        raise NotImplementedError('Not yet implemented')