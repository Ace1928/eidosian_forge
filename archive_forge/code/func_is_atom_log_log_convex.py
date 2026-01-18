import operator as op
from functools import reduce
from typing import List, Tuple
import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.utilities as u
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.constraints.constraint import Constraint
def is_atom_log_log_convex(self) -> bool:
    """Is the atom log-log convex?
        """
    return True