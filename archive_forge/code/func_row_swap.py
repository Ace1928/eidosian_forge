from types import FunctionType
from .utilities import _get_intermediate_simp, _iszero, _dotprodsimp, _simplify
from .determinant import _find_reasonable_pivot
def row_swap(i, j):
    mat[i * cols:(i + 1) * cols], mat[j * cols:(j + 1) * cols] = (mat[j * cols:(j + 1) * cols], mat[i * cols:(i + 1) * cols])