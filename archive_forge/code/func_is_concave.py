import abc
from typing import TYPE_CHECKING, List, Tuple
import numpy as np
import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
from cvxpy import interface as intf
from cvxpy import utilities as u
from cvxpy.expressions import cvxtypes
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.expression import Expression
from cvxpy.utilities import performance_utils as perf
from cvxpy.utilities.deterministic import unique_list
@perf.compute_once
def is_concave(self) -> bool:
    """Is the expression concave?
        """
    if self.is_constant():
        return True
    elif self.is_atom_concave():
        for idx, arg in enumerate(self.args):
            if not (arg.is_affine() or (arg.is_concave() and self.is_incr(idx)) or (arg.is_convex() and self.is_decr(idx))):
                return False
        return True
    else:
        return False