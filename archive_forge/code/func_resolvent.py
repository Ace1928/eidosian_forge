from typing import Tuple
import numpy as np
from cvxpy.atoms.atom import Atom
def resolvent(X, s: float):
    """The resolvent of a positive matrix, :math:`(sI - X)^{-1}`.

    For an elementwise positive matrix :math:`X` and a positive scalar
    :math:`s`, this atom computes

    .. math::

        (sI - X)^{-1},

    and it enforces the constraint that the spectral radius of :math:`X/s`
    is at most :math:`1`.

    This atom is log-log convex.

    Parameters
    ----------
    X : cvxpy.Expression
        A positive square matrix.
    s : cvxpy.Expression or numeric
        A positive scalar.
    """
    return 1.0 / s * eye_minus_inv(X / s)