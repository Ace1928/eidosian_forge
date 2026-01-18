import numpy as np
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.constraints import PSD, SOC, ExpCone, PowCone3D
from cvxpy.expressions.expression import Expression
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.utilities.versioning import Version
def clarabel_psdvec_to_psdmat(vec: Expression, indices: np.ndarray) -> Expression:
    """
    Return "V" so that "vec[indices] belongs to the Clarabel PSDTriangleCone"
    can be written in natural cvxpy syntax as "V >> 0".

    Parameters
    ----------
    vec : cvxpy.expressions.expression.Expression
        Must have ``vec.is_affine() == True``.
    indices : ndarray
        Contains nonnegative integers, which can index into ``vec``.

    Notes
    -----
    This function is similar to ``triu_to_full``, which is also found
    in this file. The difference is that this function works without
    indexed assignment ``mat[i,j] = expr``. Such indexed assignment
    cannot be used, because this function builds a cvxpy Expression,
    rather than a numpy ndarray.
    """
    n = int(np.sqrt(indices.size * 2))
    rows, cols = np.tril_indices(n)
    mats = []
    for i, idx in enumerate(indices):
        r, c = (rows[i], cols[i])
        mat = np.zeros(shape=(n, n))
        if r == c:
            mat[r, r] = 1
        else:
            mat[r, c] = 1 / np.sqrt(2)
            mat[c, r] = 1 / np.sqrt(2)
        mat = vec[idx] * mat
        mats.append(mat)
    V = sum(mats)
    return V