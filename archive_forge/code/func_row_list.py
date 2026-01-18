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
def row_list(self):
    """Returns a row-sorted list of non-zero elements of the matrix.

        Examples
        ========

        >>> from sympy import SparseMatrix
        >>> a = SparseMatrix(((1, 2), (3, 4)))
        >>> a
        Matrix([
        [1, 2],
        [3, 4]])
        >>> a.RL
        [(0, 0, 1), (0, 1, 2), (1, 0, 3), (1, 1, 4)]

        See Also
        ========

        sympy.matrices.sparse.SparseMatrix.col_list
        """
    return [tuple(k + (self[k],)) for k in sorted(self.todok().keys(), key=list)]