import operator as op
from typing import List, Tuple
import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.constraints.constraint import Constraint
def sign_from_args(self) -> Tuple[bool, bool]:
    """Returns sign (is positive, is negative) of the expression.
        """
    return (self.args[0].is_nonpos(), self.args[0].is_nonneg())