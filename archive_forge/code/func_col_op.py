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
def col_op(self, j, f):
    """In-place operation on col j using two-arg functor whose args are
        interpreted as (self[i, j], i).

        Examples
        ========

        >>> from sympy import eye
        >>> M = eye(3)
        >>> M.col_op(1, lambda v, i: v + 2*M[i, 0]); M
        Matrix([
        [1, 2, 0],
        [0, 1, 0],
        [0, 0, 1]])

        See Also
        ========
        col
        row_op
        """
    for i in range(self.rows):
        self[i, j] = f(self[i, j], i)