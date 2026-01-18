from typing import List, Tuple
import numpy as np
import cvxpy as cp
from cvxpy.atoms.affine.upper_tri import upper_tri
from cvxpy.constraints.constraint import Constraint
from cvxpy.constraints.exponential import (
from cvxpy.constraints.zero import Zero
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.dcp2cone.canonicalizers.von_neumann_entr_canon import (
def rotated_quad_cone(X: cp.Expression, y: cp.Expression, z: cp.Expression):
    """
    For each i, enforce a constraint that
        (X[i, :], y[i], z[i])
    belongs to the rotated quadratic cone
        { (x, y, z) : || x ||^2 <= y z, 0 <= (y, z) }
    This implementation doesn't enforce (x, y) >= 0! That should be imposed by the calling function.
    """
    m = y.size
    assert z.size == m
    assert X.shape[0] == m
    if len(X.shape) < 2:
        X = cp.reshape(X, (m, 1))
    soc_X_col0 = cp.reshape(y - z, (m, 1))
    soc_X = cp.hstack((soc_X_col0, 2 * X))
    soc_t = y + z
    con = cp.SOC(t=soc_t, X=soc_X, axis=1)
    return con