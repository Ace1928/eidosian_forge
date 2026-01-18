import random
import concurrent.futures
from collections.abc import Hashable
from sympy.core.add import Add
from sympy.core.function import (Function, diff, expand)
from sympy.core.numbers import (E, Float, I, Integer, Rational, nan, oo, pi)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.integrals.integrals import integrate
from sympy.polys.polytools import (Poly, PurePoly)
from sympy.printing.str import sstr
from sympy.sets.sets import FiniteSet
from sympy.simplify.simplify import (signsimp, simplify)
from sympy.simplify.trigsimp import trigsimp
from sympy.matrices.matrices import (ShapeError, MatrixError,
from sympy.matrices import (
from sympy.matrices.utilities import _dotprodsimp_state
from sympy.core import Tuple, Wild
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.utilities.iterables import flatten, capture, iterable
from sympy.utilities.exceptions import ignore_warnings, SymPyDeprecationWarning
from sympy.testing.pytest import (raises, XFAIL, slow, skip, skip_under_pyodide,
from sympy.assumptions import Q
from sympy.tensor.array import Array
from sympy.matrices.expressions import MatPow
from sympy.algebras import Quaternion
from sympy.abc import a, b, c, d, x, y, z, t
def test_fancy_index_matrix():
    for M in (Matrix, SparseMatrix):
        a = M(3, 3, range(9))
        assert a == a[:, :]
        assert a[1, :] == Matrix(1, 3, [3, 4, 5])
        assert a[:, 1] == Matrix([1, 4, 7])
        assert a[[0, 1], :] == Matrix([[0, 1, 2], [3, 4, 5]])
        assert a[[0, 1], 2] == a[[0, 1], [2]]
        assert a[2, [0, 1]] == a[[2], [0, 1]]
        assert a[:, [0, 1]] == Matrix([[0, 1], [3, 4], [6, 7]])
        assert a[0, 0] == 0
        assert a[0:2, :] == Matrix([[0, 1, 2], [3, 4, 5]])
        assert a[:, 0:2] == Matrix([[0, 1], [3, 4], [6, 7]])
        assert a[::2, 1] == a[[0, 2], 1]
        assert a[1, ::2] == a[1, [0, 2]]
        a = M(3, 3, range(9))
        assert a[[0, 2, 1, 2, 1], :] == Matrix([[0, 1, 2], [6, 7, 8], [3, 4, 5], [6, 7, 8], [3, 4, 5]])
        assert a[:, [0, 2, 1, 2, 1]] == Matrix([[0, 2, 1, 2, 1], [3, 5, 4, 5, 4], [6, 8, 7, 8, 7]])
    a = SparseMatrix.zeros(3)
    a[1, 2] = 2
    a[0, 1] = 3
    a[2, 0] = 4
    assert a.extract([1, 1], [2]) == Matrix([[2], [2]])
    assert a.extract([1, 0], [2, 2, 2]) == Matrix([[2, 2, 2], [0, 0, 0]])
    assert a.extract([1, 0, 1, 2], [2, 0, 1, 0]) == Matrix([[2, 0, 0, 0], [0, 0, 3, 0], [2, 0, 0, 0], [0, 4, 0, 4]])