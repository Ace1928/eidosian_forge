from types import FunctionType
from .utilities import _get_intermediate_simp, _iszero, _dotprodsimp, _simplify
from .determinant import _find_reasonable_pivot
def complexity(i):
    return sum((1 if iszerofunc(e) is None else 0 for e in M[:, i]))