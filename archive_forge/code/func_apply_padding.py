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
def apply_padding(e1_var: TVar, e11: BinConstraintT, e2: BinConstraintT, e12: BinConstraintT, d2: List[DVar], d11: List[DVar], d12: List[DVar], counter: int):
    """
    We are considering the possibility where one input has less dimensions than
    another input, so we apply padding to the broadcasted results

    Args:
        e1_var: Variable representing the first input where padding will be
        e11: constraint of the form e11 = Tensortype[d1, ..., dn]
        e2:  constraint of the form e2 = Tensortype[d1, ..., dn]
        e12: constraint of the form e11 = Tensortype[d1, ..., dn]
        d2: Tensor variables for the second input
        d11: Tensor variables for the broadcasted first input
        d12: Tensor variables for the broadcasted second input
        counter: variable tracking

    Returns: A new constraint whose goal is to apply padding to the broadcasted result

    """
    res = []
    for i in range(1, len(d2)):
        d1, counter = gen_tensor_dims(i, counter)
        nat_constraints = gen_nat_constraints(d1 + d2 + d11 + d12)
        e1 = BinConstraintT(e1_var, TensorType(d1), op_eq)
        simulate_padding = [None] * (len(d2) - i)
        assert len(simulate_padding + d1) == len(d2)
        broadcast_padding = []
        for j in range(len(d2) - i):
            broadcast_padding.append(broadcast_dim(simulate_padding, d2, d11, d12, j, True))
        all_broadcasting_possibilities = generate_all_broadcasting_possibilities_no_padding(d1, d2[len(d2) - i:], d11[len(d2) - i:], d12[len(d2) - i:])
        c = Conj([e1, e11, e2, e12, *broadcast_padding, all_broadcasting_possibilities, *nat_constraints])
        res.append(c)
    return (Disj(res), counter)