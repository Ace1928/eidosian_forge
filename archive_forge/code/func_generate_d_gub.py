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
@register_transformation_rule(DGreatestUpperBound)
def generate_d_gub(constraint, counter):
    """
    Transform greatest upper bound for dimensions into equality constraints
    """
    c1 = Conj([BinConstraintD(constraint.rhs1, Dyn, op_eq), BinConstraintD(constraint.res, constraint.rhs2, op_eq)])
    c2 = Conj([BinConstraintD(constraint.rhs2, Dyn, op_eq), BinConstraintD(constraint.res, constraint.rhs1, op_eq)])
    c3 = Conj([BinConstraintD(constraint.rhs2, constraint.rhs1, op_eq), BinConstraintD(constraint.res, constraint.rhs1, op_eq)])
    return (Disj([c1, c2, c3]), counter)