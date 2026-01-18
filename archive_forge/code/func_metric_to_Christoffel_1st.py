from __future__ import annotations
from typing import Any
from functools import reduce
from itertools import permutations
from sympy.combinatorics import Permutation
from sympy.core import (
from sympy.core.cache import cacheit
from sympy.core.symbol import Symbol, Dummy
from sympy.core.symbol import Str
from sympy.core.sympify import _sympify
from sympy.functions import factorial
from sympy.matrices import ImmutableDenseMatrix as Matrix
from sympy.solvers import solve
from sympy.utilities.exceptions import (sympy_deprecation_warning,
from sympy.tensor.array import ImmutableDenseNDimArray
from sympy.simplify.simplify import simplify
def metric_to_Christoffel_1st(expr):
    """Return the nested list of Christoffel symbols for the given metric.
    This returns the Christoffel symbol of first kind that represents the
    Levi-Civita connection for the given metric.

    Examples
    ========

    >>> from sympy.diffgeom.rn import R2
    >>> from sympy.diffgeom import metric_to_Christoffel_1st, TensorProduct
    >>> TP = TensorProduct

    >>> metric_to_Christoffel_1st(TP(R2.dx, R2.dx) + TP(R2.dy, R2.dy))
    [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
    >>> metric_to_Christoffel_1st(R2.x*TP(R2.dx, R2.dx) + TP(R2.dy, R2.dy))
    [[[1/2, 0], [0, 0]], [[0, 0], [0, 0]]]

    """
    matrix = twoform_to_matrix(expr)
    if not matrix.is_symmetric():
        raise ValueError('The two-form representing the metric is not symmetric.')
    coord_sys = _find_coords(expr).pop()
    deriv_matrices = [matrix.applyfunc(d) for d in coord_sys.base_vectors()]
    indices = list(range(coord_sys.dim))
    christoffel = [[[(deriv_matrices[k][i, j] + deriv_matrices[j][i, k] - deriv_matrices[i][j, k]) / 2 for k in indices] for j in indices] for i in indices]
    return ImmutableDenseNDimArray(christoffel)