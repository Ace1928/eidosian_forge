import copy
import itertools
from torch.fx.experimental.migrate_gradual_types.constraint_generator import BinConstraintT, MAX_TENSOR_RANK
from torch.fx.experimental.migrate_gradual_types.constraint import T, BinConstraintD, Conj, Constraint, DVar, TVar, \
from torch.fx.experimental.migrate_gradual_types.constraint import Disj, TGreatestUpperBound
from torch.fx.experimental.migrate_gradual_types.constraint import DGreatestUpperBound
from torch.fx.experimental.migrate_gradual_types.constraint import CalcConv, CalcMaxPool
from torch.fx.experimental.migrate_gradual_types.constraint import CalcProduct, CanReshape
from torch.fx.experimental.migrate_gradual_types.constraint import ApplyBroadcasting, Prod, F, GetItem, GetItemTensor, IndexSelect
from torch.fx.experimental.migrate_gradual_types.operation import op_eq, op_precision, op_leq, op_matching
from torch.fx.experimental.migrate_gradual_types.operation import op_consistency, op_neq
from torch.fx.experimental.migrate_gradual_types.operation import op_mul, op_add, op_sub, op_div, op_mod
from torch.fx.experimental.migrate_gradual_types.util import gen_tensor_dims, gen_nat_constraints, gen_dvar
from torch.fx.tensor_type import TensorType, Dyn
from typing import Callable, Dict, List
def gen_consistency_constraints(constraint: Constraint, counter: int):
    """
    Args:
        constraint: Consistency constraint on tensors
        counter: for variable tracking

    Returns: Equality and consistency constraints on dimensions

    """
    all_constraints = []
    for i in range(1, MAX_TENSOR_RANK + 1):
        new_dims_rhs_1, counter = gen_tensor_dims(i, counter)
        new_dims_rhs_2, counter = gen_tensor_dims(i, counter)
        nat_constraints = gen_nat_constraints(new_dims_rhs_1 + new_dims_rhs_2)
        c_tensor_i = Conj([BinConstraintT(constraint.lhs, TensorType(new_dims_rhs_1), op_eq), BinConstraintT(constraint.rhs, TensorType(new_dims_rhs_2), op_eq)] + [BinConstraintD(d1, d2, op_consistency) for d1, d2 in zip(new_dims_rhs_1, new_dims_rhs_2)] + nat_constraints)
        all_constraints.append(c_tensor_i)
    return (all_constraints, counter)