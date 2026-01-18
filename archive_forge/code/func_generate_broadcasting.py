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
@register_transformation_rule(ApplyBroadcasting)
def generate_broadcasting(constraint, counter):
    """
    Transform broadcasting constraints
    """
    e11, e12 = (constraint.res1, constraint.res2)
    e1, e2 = (constraint.input1, constraint.input2)
    e1_dyn = BinConstraintT(e1, Dyn, op_eq)
    e2_dyn = BinConstraintT(e2, Dyn, op_eq)
    e1_equal_e11 = BinConstraintT(e1, e11, op_eq)
    e2_equal_e12 = BinConstraintT(e2, e12, op_eq)
    e1_dyn_constraint = Conj([e1_dyn, e1_equal_e11, e2_equal_e12])
    e2_dyn_constraint = Conj([e2_dyn, e1_equal_e11, e2_equal_e12])
    final_tensor_1_constraint, _, _, nat_dims_1, counter = gen_broadcasting_constraints(e1, e2, e11, e12, 1, counter)
    final_tensor_2_constraint_no_padding, final_tensor_2_constraint_padding_arg1, final_tensor_2_constraint_padding_arg2, nat_dims_2, counter = gen_broadcasting_constraints(e1, e2, e11, e12, 2, counter)
    final_tensor_3_constraint_no_padding, final_tensor_3_constraint_padding_arg1, final_tensor_3_constraint_padding_arg2, nat_dims_3, counter = gen_broadcasting_constraints(e1, e2, e11, e12, 3, counter)
    final_tensor_4_constraint_no_padding, final_tensor_4_constraint_padding_arg1, final_tensor_4_constraint_padding_arg2, nat_dims_4, counter = gen_broadcasting_constraints(e1, e2, e11, e12, 4, counter)
    final_result = Disj([e1_dyn_constraint, e2_dyn_constraint, final_tensor_1_constraint, final_tensor_2_constraint_no_padding, final_tensor_2_constraint_padding_arg1, final_tensor_2_constraint_padding_arg2, final_tensor_3_constraint_no_padding, final_tensor_3_constraint_padding_arg1, final_tensor_3_constraint_padding_arg2, final_tensor_4_constraint_no_padding, final_tensor_4_constraint_padding_arg1, final_tensor_4_constraint_padding_arg2])
    return (Conj([final_result, *nat_dims_1, *nat_dims_2, *nat_dims_3, *nat_dims_4]), counter)