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
def gen_embedding_rules(n: Node, symbols, embedding_dim, counter):
    embedding_output, counter = gen_tvar(counter)
    symbols[n] = embedding_output
    embedding_input = symbols[n.args[0]]
    input_dyn = BinConstraintT(embedding_input, Dyn, op_eq)
    output_dyn = BinConstraintT(embedding_output, Dyn, op_eq)
    c1 = Conj([input_dyn, output_dyn])
    c2 = []
    for i in range(1, MAX_TENSOR_RANK):
        new_dims, counter = gen_tensor_dims(i, counter)
        nat_constraints = gen_nat_constraints(new_dims)
        c_tensor_i = Conj([BinConstraintT(embedding_input, TensorType(new_dims), op_eq), BinConstraintT(embedding_output, TensorType(new_dims + [embedding_dim]), op_eq)] + nat_constraints)
        c2.append(c_tensor_i)
    return ([Disj([c1, Disj(c2)])], counter)