from sympy.assumptions import Q
from sympy.core.relational import is_eq, is_neq, is_gt, is_ge, is_lt, is_le
from .binrel import BinaryRelation
@property
def negated(self):
    return Q.gt