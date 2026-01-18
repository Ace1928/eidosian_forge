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
@register_transformation_rule(CalcConv)
def generate_calc_conv(constraint, counter):
    d, counter = gen_tensor_dims(4, counter)
    conv_result = TensorType([d[0], d[1], d[2], d[3]])
    c1 = BinConstraintT(constraint.conv_result, conv_result, op_eq)
    c2 = Conj([BinConstraintD(d[1], constraint.c_out, op_eq), BinConstraintD(d[1], Dyn, op_neq)])
    c3 = BinConstraintD(constraint.matching_constraint[0], d[0], op_eq)
    c4, c5 = calc_last_two_dims(constraint, d)
    leq_constraints = Conj([BinConstraintD(0, d[0], op_leq), BinConstraintD(0, d[1], op_leq), BinConstraintD(0, d[2], op_leq), BinConstraintD(0, d[3], op_leq)])
    return (Conj([c1, c2, c3, c4, c5, leq_constraints]), counter)