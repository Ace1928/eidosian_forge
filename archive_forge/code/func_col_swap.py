from collections import defaultdict
from operator import index as index_
from sympy.core.expr import Expr
from sympy.core.kind import Kind, NumberKind, UndefinedKind
from sympy.core.numbers import Integer, Rational
from sympy.core.sympify import _sympify, SympifyError
from sympy.core.singleton import S
from sympy.polys.domains import ZZ, QQ, EXRAW
from sympy.polys.matrices import DomainMatrix
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import filldedent
from .common import classof
from .matrices import MatrixBase, MatrixKind, ShapeError
def col_swap(self, i, j):
    """Swap the two given columns of the matrix in-place.

        Examples
        ========

        >>> from sympy import Matrix
        >>> M = Matrix([[1, 0], [1, 0]])
        >>> M
        Matrix([
        [1, 0],
        [1, 0]])
        >>> M.col_swap(0, 1)
        >>> M
        Matrix([
        [0, 1],
        [0, 1]])

        See Also
        ========

        col
        row_swap
        """
    for k in range(0, self.rows):
        self[k, i], self[k, j] = (self[k, j], self[k, i])