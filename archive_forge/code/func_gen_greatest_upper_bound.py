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
def gen_greatest_upper_bound(constraint: TGreatestUpperBound, counter: int):
    """
    Args:
        constraint: Greatest upper bound on tensors
        counter: variable tracking

    Returns: A set of equality constraints and DGreatestUpperBound constraints

    """
    all_constraints = []
    for i in range(1, MAX_TENSOR_RANK + 1):
        c = []
        dims1, counter = gen_tensor_dims(i, counter)
        c1tensor = TensorType(dims1)
        dims2, counter = gen_tensor_dims(i, counter)
        c2tensor = TensorType(dims2)
        dims3, counter = gen_tensor_dims(i, counter)
        c3tensor = TensorType(dims3)
        c += [BinConstraintT(constraint.rhs1, c1tensor, op_eq), BinConstraintT(constraint.rhs2, c2tensor, op_eq), BinConstraintT(constraint.res, c3tensor, op_eq)] + gen_nat_constraints(dims1 + dims2 + dims3)
        assert len(c3tensor.__args__) == len(c1tensor.__args__) == len(c2tensor.__args__)
        for i in range(len(c3tensor.__args__)):
            c.append(DGreatestUpperBound(c3tensor.__args__[i], c1tensor.__args__[i], c2tensor.__args__[i]))
        all_constraints.append(Conj(c))
    return (all_constraints, counter)