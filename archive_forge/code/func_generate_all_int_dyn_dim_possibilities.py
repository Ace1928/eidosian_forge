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
def generate_all_int_dyn_dim_possibilities(my_list: List[DVar]):
    """
    Generate all possibilities of being equal or not equal to dyn for my_list
    Args:
        my_list: List of tensor dimensions

    Returns: A list of a list of constraints. Each list of constraints corresponds to
    one possibility about the values of the dimension variables
    """
    eq_possibilities = [BinConstraintD(my_list[i], Dyn, op_eq) for i in range(len(my_list))]
    neq_possibilities = [BinConstraintD(my_list[i], Dyn, op_neq) for i in range(len(my_list))]
    d_possibilities = []
    for i in zip(eq_possibilities, neq_possibilities):
        d_possibilities.append(list(i))
    all_possibilities = list(itertools.product(*d_possibilities))
    return all_possibilities