from __future__ import annotations
import cvxpy.lin_ops.lin_utils as lu
from cvxpy import settings as s
from cvxpy.expressions.leaf import Leaf
from cvxpy.utilities import scopes
def is_param_free(expr) -> bool:
    """Returns true if expression is not parametrized."""
    return not expr.parameters()