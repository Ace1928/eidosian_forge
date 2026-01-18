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
@register_inference_rule(torch.nn.LayerNorm)
def layer_norm_inference_rule(n: Node, module_instance, symbols, constraints, counter):
    """
    Input and output shapes should be equal.
    Input should be consistent with the normalized_shape
    """
    assert isinstance(n.args[0], Node)
    return gen_layer_norm_constraints(n, module_instance.normalized_shape, symbols, counter)