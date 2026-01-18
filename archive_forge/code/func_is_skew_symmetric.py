from typing import List, Tuple
import numpy as np
import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.constraints.constraint import Constraint
def is_skew_symmetric(self) -> bool:
    """Is the expression skew-symmetric?
        """
    return self.args[0].is_skew_symmetric()