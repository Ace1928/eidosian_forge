from collections import defaultdict
from collections.abc import Iterable
from inspect import isfunction
from functools import reduce
from sympy.assumptions.refine import refine
from sympy.core import SympifyError, Add
from sympy.core.basic import Atom
from sympy.core.decorators import call_highest_priority
from sympy.core.kind import Kind, NumberKind
from sympy.core.logic import fuzzy_and, FuzzyBool
from sympy.core.mod import Mod
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import Abs, re, im
from .utilities import _dotprodsimp, _simplify
from sympy.polys.polytools import Poly
from sympy.utilities.iterables import flatten, is_sequence
from sympy.utilities.misc import as_int, filldedent
from sympy.tensor.array import NDimArray
from .utilities import _get_intermediate_simp_bool
@property
def is_strongly_diagonally_dominant(self):
    """Tests if the matrix is row strongly diagonally dominant.

        Explanation
        ===========

        A $n, n$ matrix $A$ is row strongly diagonally dominant if

        .. math::
            \\left|A_{i, i}\\right| > \\sum_{j = 0, j \\neq i}^{n-1}
            \\left|A_{i, j}\\right| \\quad {\\text{for all }}
            i \\in \\{ 0, ..., n-1 \\}

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix([[3, -2, 1], [1, -3, 2], [-1, 2, 4]])
        >>> A.is_strongly_diagonally_dominant
        False

        >>> A = Matrix([[-2, 2, 1], [1, 3, 2], [1, -2, 0]])
        >>> A.is_strongly_diagonally_dominant
        False

        >>> A = Matrix([[-4, 2, 1], [1, 6, 2], [1, -2, 5]])
        >>> A.is_strongly_diagonally_dominant
        True

        Notes
        =====

        If you want to test whether a matrix is column diagonally
        dominant, you can apply the test after transposing the matrix.
        """
    if not self.is_square:
        return False
    rows, cols = self.shape

    def test_row(i):
        summation = self.zero
        for j in range(cols):
            if i != j:
                summation += Abs(self[i, j])
        return (Abs(self[i, i]) - summation).is_positive
    return fuzzy_and((test_row(i) for i in range(rows)))