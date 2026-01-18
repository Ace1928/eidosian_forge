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
def row_op(self, i, f):
    """In-place operation on row ``i`` using two-arg functor whose args are
        interpreted as ``(self[i, j], j)``.

        Examples
        ========

        >>> from sympy import eye
        >>> M = eye(3)
        >>> M.row_op(1, lambda v, j: v + 2*M[0, j]); M
        Matrix([
        [1, 0, 0],
        [2, 1, 0],
        [0, 0, 1]])

        See Also
        ========
        row
        zip_row_op
        col_op

        """
    for j in range(self.cols):
        self[i, j] = f(self[i, j], j)