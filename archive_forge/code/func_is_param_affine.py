from __future__ import annotations
import cvxpy.lin_ops.lin_utils as lu
from cvxpy import settings as s
from cvxpy.expressions.leaf import Leaf
from cvxpy.utilities import scopes
def is_param_affine(expr) -> bool:
    """Returns true if expression is parameters-affine (and variable-free)"""
    with scopes.dpp_scope():
        return not expr.variables() and expr.is_affine()