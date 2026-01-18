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
@register_inference_rule(operator.ne)
def neq_inference_rule(n: Node, symbols, constraints, counter):
    """
    Translates to inconsistent in gradual types.
    To prove inequality, we should prove that
    tensors are either different sizes or
    disagree on at least one dimension

    This is a WIP (works when the condition
    is false. We are working on making this operation work
    when the condition is true as well)
    """
    assert isinstance(n.args[0], Node)
    assert isinstance(n.args[1], tuple)
    if len(n.args[1]) == 3:
        assert isinstance(n.args[1][0], (Node, int))
        assert isinstance(n.args[1][1], (Node, int))
        assert isinstance(n.args[1][2], (Node, int))
        lhs = symbols[n.args[0]]
        b, counter = gen_tensor_dims(4, counter)
        input_is_size3 = BinConstraintT(lhs, TensorType([b[0], b[1], b[2]]), op_eq)
        d1 = n.args[1][0] if isinstance(n.args[1][0], int) else symbols[n.args[1][0]]
        d2 = n.args[1][1] if isinstance(n.args[1][1], int) else symbols[n.args[1][1]]
        d3 = n.args[1][2] if isinstance(n.args[1][2], int) else symbols[n.args[1][2]]
        my_ne, counter = gen_bvar(counter)
        neq_1 = BinConstraintD(d1, b[0], op_neq)
        neq_2 = BinConstraintD(d2, b[1], op_neq)
        neq_3 = BinConstraintD(d3, b[2], op_neq)
        dims_inconsistent1 = Conj([BinConstraintD(d1, Dyn, op_neq), BinConstraintD(b[0], Dyn, op_neq), neq_1])
        dims_inconsistent2 = Conj([BinConstraintD(d2, Dyn, op_neq), BinConstraintD(b[1], Dyn, op_neq), neq_2])
        dims_inconsistent3 = Conj([BinConstraintD(d3, Dyn, op_neq), BinConstraintD(b[2], Dyn, op_neq), neq_3])
        dims_inconsistent = Disj([dims_inconsistent1, dims_inconsistent2, dims_inconsistent3])
        ne_constraint = Conj([input_is_size3, dims_inconsistent])
        my_ne, counter = gen_bvar(counter)
        equality_constraint = BinConstraintD(my_ne, ne_constraint, op_eq)
    elif len(n.args[1]) == 4:
        assert isinstance(n.args[1][0], (Node, int))
        assert isinstance(n.args[1][1], (Node, int))
        assert isinstance(n.args[1][2], (Node, int))
        assert isinstance(n.args[1][3], (Node, int))
        lhs = symbols[n.args[0]]
        b1, counter = gen_dvar(counter)
        b2, counter = gen_dvar(counter)
        b3, counter = gen_dvar(counter)
        b4, counter = gen_dvar(counter)
        input_is_size4 = BinConstraintT(lhs, TensorType([b1, b2, b3, b4]), op_eq)
        d1 = n.args[1][0] if isinstance(n.args[1][0], int) else symbols[n.args[1][0]]
        d2 = n.args[1][1] if isinstance(n.args[1][1], int) else symbols[n.args[1][1]]
        d3 = n.args[1][2] if isinstance(n.args[1][2], int) else symbols[n.args[1][2]]
        d4 = n.args[1][3] if isinstance(n.args[1][3], int) else symbols[n.args[1][3]]
        my_ne, counter = gen_bvar(counter)
        neq_1 = BinConstraintD(d1, b1, op_neq)
        neq_2 = BinConstraintD(d2, b2, op_neq)
        neq_3 = BinConstraintD(d3, b3, op_neq)
        neq_4 = BinConstraintD(d4, b4, op_neq)
        dims_inconsistent1 = Conj([BinConstraintD(d1, Dyn, op_neq), BinConstraintD(b1, Dyn, op_neq), neq_1])
        dims_inconsistent2 = Conj([BinConstraintD(d2, Dyn, op_neq), BinConstraintD(b2, Dyn, op_neq), neq_2])
        dims_inconsistent3 = Conj([BinConstraintD(d3, Dyn, op_neq), BinConstraintD(b3, Dyn, op_neq), neq_3])
        dims_inconsistent4 = Conj([BinConstraintD(d4, Dyn, op_neq), BinConstraintD(b3, Dyn, op_neq), neq_4])
        dims_inconsistent = Disj([dims_inconsistent1, dims_inconsistent2, dims_inconsistent3, dims_inconsistent4])
        ne_constraint = Conj([input_is_size4, dims_inconsistent])
        my_ne, counter = gen_bvar(counter)
        equality_constraint = BinConstraintD(my_ne, ne_constraint, op_eq)
    else:
        raise NotImplementedError('Method not yet implemented')
    return ([equality_constraint], counter)