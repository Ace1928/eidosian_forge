import operator as op
from typing import List, Tuple
import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.constraints.constraint import Constraint
def is_decr(self, idx) -> bool:
    """Is the composition non-increasing in argument idx?
        """
    return True