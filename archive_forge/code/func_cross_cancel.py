from types import FunctionType
from .utilities import _get_intermediate_simp, _iszero, _dotprodsimp, _simplify
from .determinant import _find_reasonable_pivot
def cross_cancel(a, i, b, j):
    """Does the row op row[i] = a*row[i] - b*row[j]"""
    q = (j - i) * cols
    for p in range(i * cols, (i + 1) * cols):
        mat[p] = isimp(a * mat[p] - b * mat[p + q])