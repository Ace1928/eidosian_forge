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
@register_inference_rule(Conv2d)
def conv2d_inference_rule(n: Node, module_instance, symbols, constraints, counter):
    assert isinstance(n.args[0], Node)
    my_conv, counter = gen_tvar(counter)
    symbols[n] = my_conv
    input_var = symbols[n.args[0]]
    [d1, d2, d3, d4], counter = gen_tensor_dims(MAX_TENSOR_RANK, counter)
    c1 = BinConstraintT(input_var, TensorType([d1, d2, d3, d4]), op_matching)
    c2 = BinConstraintD(module_instance.in_channels, d2, op_consistency)
    c3 = CalcConv(my_conv, input_var, module_instance.out_channels, module_instance.kernel_size, module_instance.padding, module_instance.stride, module_instance.dilation, [d1, d2, d3, d4])
    nat_constraints = gen_nat_constraints([d1, d2, d3, d4])
    return ([c1, c2, c3, *nat_constraints], counter)