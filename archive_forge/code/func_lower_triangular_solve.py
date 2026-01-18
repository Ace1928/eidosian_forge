from collections.abc import Callable
from sympy.core.containers import Dict
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import as_int
from .matrices import MatrixBase
from .repmatrix import MutableRepMatrix, RepMatrix
from .utilities import _iszero
from .decompositions import (
from .solvers import (
def lower_triangular_solve(self, rhs):
    return _lower_triangular_solve_sparse(self, rhs)