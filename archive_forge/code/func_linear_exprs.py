from __future__ import absolute_import, division, print_function
from itertools import chain
import numpy as np
from sym import Backend
from sym.util import banded_jacobian, check_transforms
from .core import NeqSys, _ensure_3args
def linear_exprs(A, x, b=None, rref=False, Matrix=None):
    """ Returns Ax - b

    Parameters
    ----------
    A : matrix_like of numbers
        Of shape (len(b), len(x)).
    x : iterable of symbols
    b : array_like of numbers (default: None)
        When ``None``, assume zeros of length ``len(x)``.
    Matrix : class
        When ``rref == True``: A matrix class which supports slicing,
        and methods ``__mul__`` and ``rref``. Defaults to ``sympy.Matrix``.
    rref : bool
        Calculate the reduced row echelon form of (A | -b).

    Returns
    -------
    A list of the elements in the resulting column vector.

    """
    if b is None:
        b = [0] * len(x)
    if rref:
        rA, rb = linear_rref(A, b, Matrix)
        if Matrix is None:
            from sympy import Matrix
        return [lhs - rhs for lhs, rhs in zip(rA * Matrix(len(x), 1, x), rb)]
    else:
        return [sum([x0 * x1 for x0, x1 in zip(row, x)]) - v for row, v in zip(A, b)]