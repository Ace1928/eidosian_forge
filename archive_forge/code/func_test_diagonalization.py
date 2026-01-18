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
def test_diagonalization():
    m = Matrix([[1, 2 + I], [2 - I, 3]])
    assert m.is_diagonalizable()
    m = Matrix(3, 2, [-3, 1, -3, 20, 3, 10])
    assert not m.is_diagonalizable()
    assert not m.is_symmetric()
    raises(NonSquareMatrixError, lambda: m.diagonalize())
    m = diag(1, 2, 3)
    P, D = m.diagonalize()
    assert P == eye(3)
    assert D == m
    m = Matrix(2, 2, [0, 1, 1, 0])
    assert m.is_symmetric()
    assert m.is_diagonalizable()
    P, D = m.diagonalize()
    assert P.inv() * m * P == D
    m = Matrix(2, 2, [1, 0, 0, 3])
    assert m.is_symmetric()
    assert m.is_diagonalizable()
    P, D = m.diagonalize()
    assert P.inv() * m * P == D
    assert P == eye(2)
    assert D == m
    m = Matrix(2, 2, [1, 1, 0, 0])
    assert m.is_diagonalizable()
    P, D = m.diagonalize()
    assert P.inv() * m * P == D
    m = Matrix(3, 3, [1, 2, 0, 0, 3, 0, 2, -4, 2])
    assert m.is_diagonalizable()
    P, D = m.diagonalize()
    assert P.inv() * m * P == D
    for i in P:
        assert i.as_numer_denom()[1] == 1
    m = Matrix(2, 2, [1, 0, 0, 0])
    assert m.is_diagonal()
    assert m.is_diagonalizable()
    P, D = m.diagonalize()
    assert P.inv() * m * P == D
    assert P == Matrix([[0, 1], [1, 0]])
    m = Matrix(2, 2, [0, 1, -1, 0])
    assert not m.is_diagonalizable(True)
    raises(MatrixError, lambda: m.diagonalize(True))
    assert m.is_diagonalizable()
    P, D = m.diagonalize()
    assert P.inv() * m * P == D
    m = Matrix(2, 2, [0, 1, 0, 0])
    assert not m.is_diagonalizable()
    raises(MatrixError, lambda: m.diagonalize())
    m = Matrix(3, 3, [-3, 1, -3, 20, 3, 10, 2, -2, 4])
    assert not m.is_diagonalizable()
    raises(MatrixError, lambda: m.diagonalize())
    a, b, c, d = symbols('a b c d')
    m = Matrix(2, 2, [a, c, c, b])
    assert m.is_symmetric()
    assert m.is_diagonalizable()