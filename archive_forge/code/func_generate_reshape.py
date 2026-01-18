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
@register_transformation_rule(CanReshape)
def generate_reshape(constraint, counter):
    """
    Transform reshape constraints
    """
    d, counter = gen_tensor_dims(4, counter)
    d1 = d[0]
    d2 = d[1]
    d3 = d[2]
    d4 = d[3]
    target = constraint.target.__args__
    is_fully_static = all((d != Dyn for d in target))
    c1_dyn = BinConstraintT(constraint.src, Dyn, op_eq)
    c2_tensor1 = BinConstraintT(constraint.src, TensorType([d1]), op_eq)
    c2_tensor2 = BinConstraintT(constraint.src, TensorType([d1, d2]), op_eq)
    c2_tensor3 = BinConstraintT(constraint.src, TensorType([d1, d2, d3]), op_eq)
    c2_tensor4 = BinConstraintT(constraint.src, TensorType([d1, d2, d3, d4]), op_eq)
    d1_eq_dyn = BinConstraintD(d1, Dyn, op_eq)
    d1_neq_dyn = BinConstraintD(d1, Dyn, op_neq)
    d2_eq_dyn = BinConstraintD(d2, Dyn, op_eq)
    d2_neq_dyn = BinConstraintD(d2, Dyn, op_neq)
    d3_eq_dyn = BinConstraintD(d3, Dyn, op_eq)
    d3_neq_dyn = BinConstraintD(d3, Dyn, op_neq)
    d4_eq_dyn = BinConstraintD(d3, Dyn, op_eq)
    d4_neq_dyn = BinConstraintD(d3, Dyn, op_neq)
    nat_d1 = BinConstraintD(0, d1, op_leq)
    nat_d2 = BinConstraintD(0, d2, op_leq)
    nat_d3 = BinConstraintD(0, d3, op_leq)
    nat_d4 = BinConstraintD(0, d4, op_leq)
    if is_fully_static:
        c3_tensor1 = Disj([d1_eq_dyn, Conj([d1_neq_dyn, BinConstraintD(d1, Prod(target), op_eq)])])
        all_tensor_1 = Conj([c2_tensor1, c3_tensor1])
        all_tensor_2 = Conj([c2_tensor2, gen_all_reshape_possibilities([d1, d2], target)])
        all_tensor_3 = Conj([c2_tensor3, gen_all_reshape_possibilities([d1, d2, d3], target)])
        all_tensor_4 = Conj([c2_tensor4, gen_all_reshape_possibilities([d1, d2, d3, d4], target)])
        return (Conj([Disj([c1_dyn, all_tensor_1, all_tensor_2, all_tensor_3, all_tensor_4]), nat_d1, nat_d2, nat_d3, nat_d4]), counter)
    else:
        new_target = []
        for n in target:
            if n != Dyn:
                new_target.append(n)
        c3_tensor1 = Disj([d1_eq_dyn, Conj([d1_neq_dyn, is_dim_div_by_target(new_target, d1)])])
        all_tensor_1 = Conj([c2_tensor1, c3_tensor1])
        c21 = Disj([d1_eq_dyn, d2_eq_dyn])
        c22 = Conj([d1_neq_dyn, d2_neq_dyn, is_dim_div_by_target(new_target, Prod([d1, d2]))])
        all_tensor_2 = Conj([c2_tensor2, Disj([c21, c22])])
        c31 = Disj([d1_eq_dyn, d2_eq_dyn, d3_eq_dyn])
        c32 = Conj([d1_neq_dyn, d2_neq_dyn, d3_neq_dyn, is_dim_div_by_target(new_target, Prod([d1, d2, d3]))])
        all_tensor_3 = Conj([c2_tensor3, Disj([c31, c32])])
        c41 = Disj([d1_eq_dyn, d2_eq_dyn, d3_eq_dyn, d4_eq_dyn])
        c42 = Conj([d1_neq_dyn, d2_neq_dyn, d3_neq_dyn, d4_neq_dyn, is_dim_div_by_target(new_target, Prod([d1, d2, d3, d4]))])
        all_tensor_4 = Conj([c2_tensor4, Disj([c41, c42])])
        return (Conj([Disj([c1_dyn, all_tensor_1, all_tensor_2, all_tensor_3, all_tensor_4]), nat_d1, nat_d2, nat_d3, nat_d4]), counter)