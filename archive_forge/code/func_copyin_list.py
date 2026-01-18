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
def copyin_list(self, key, value):
    """Copy in elements from a list.

        Parameters
        ==========

        key : slice
            The section of this matrix to replace.
        value : iterable
            The iterable to copy values from.

        Examples
        ========

        >>> from sympy import eye
        >>> I = eye(3)
        >>> I[:2, 0] = [1, 2] # col
        >>> I
        Matrix([
        [1, 0, 0],
        [2, 1, 0],
        [0, 0, 1]])
        >>> I[1, :2] = [[3, 4]]
        >>> I
        Matrix([
        [1, 0, 0],
        [3, 4, 0],
        [0, 0, 1]])

        See Also
        ========

        copyin_matrix
        """
    if not is_sequence(value):
        raise TypeError('`value` must be an ordered iterable, not %s.' % type(value))
    return self.copyin_matrix(key, type(self)(value))