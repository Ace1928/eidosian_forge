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
def generate_all_broadcasting_possibilities_no_padding(d1: List[DVar], d2: List[DVar], d11: List[DVar], d12: List[DVar]):
    """
    Generate broadcasting constraints assuming no padding. Broadcasting can happen at any dimension.
    We look at all combinations for all dimensions in d1 and d2
    Args:
        d1: input1 dimensions
        d2: input2 dimensions
        d11: broadcasted input1 dimensions
        d12: broadcasted input2 dimensions

    Returns: broadcasting constraints relating the input dimensions to the broadcasted dimensions

    """
    size = len(d1)
    res2 = []
    for i in range(size):
        t1 = broadcast_dim(d1, d2, d11, d12, i)
        t2 = broadcast_dim(d2, d1, d12, d11, i)
        t3 = no_broadcast_dim_with_index(d1, d2, d11, d12, i)
        res2.append(Disj([t1, t2, t3]))
    return Conj(res2)