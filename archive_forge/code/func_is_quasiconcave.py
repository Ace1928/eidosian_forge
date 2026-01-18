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
def is_quasiconcave(self) -> bool:
    """Is the expression quasiconcave?
        """
    from cvxpy.atoms.min import min as min_atom
    if self.is_concave():
        return True
    if type(self) in (cvxtypes.minimum(), min_atom):
        return all((arg.is_quasiconcave() for arg in self.args))
    non_const = self._non_const_idx()
    if self._is_real() and self.is_incr(non_const[0]):
        return self.args[non_const[0]].is_quasiconcave()
    if self._is_real() and self.is_decr(non_const[0]):
        return self.args[non_const[0]].is_quasiconvex()
    if self.is_atom_quasiconcave():
        for idx, arg in enumerate(self.args):
            if not (arg.is_affine() or (arg.is_concave() and self.is_incr(idx)) or (arg.is_convex() and self.is_decr(idx))):
                return False
        return True
    return False