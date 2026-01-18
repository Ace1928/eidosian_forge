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
@register_inference_rule(torch.nn.functional.embedding)
def embedding_inference_rule_functional(n: Node, symbols, constraints, counter):
    assert isinstance(n.args[0], Node)
    embedding_dim_weights = symbols[n.args[1]]
    weight_dims, counter = gen_tensor_dims(2, counter)
    equality_constraint = BinConstraintT(embedding_dim_weights, TensorType(weight_dims), op_eq)
    embedding_dim = weight_dims[1]
    constraints, counter = gen_embedding_rules(n, symbols, embedding_dim, counter)
    return ([equality_constraint] + constraints, counter)