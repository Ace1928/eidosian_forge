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
def test_diff_by_matrix():
    A = MutableDenseMatrix([[x, y], [z, t]])
    assert A.diff(A) == Array([[[[1, 0], [0, 0]], [[0, 1], [0, 0]]], [[[0, 0], [1, 0]], [[0, 0], [0, 1]]]])
    assert diff(A, A) == Array([[[[1, 0], [0, 0]], [[0, 1], [0, 0]]], [[[0, 0], [1, 0]], [[0, 0], [0, 1]]]])
    A_imm = A.as_immutable()
    assert A_imm.diff(A_imm) == Array([[[[1, 0], [0, 0]], [[0, 1], [0, 0]]], [[[0, 0], [1, 0]], [[0, 0], [0, 1]]]])
    assert diff(A_imm, A_imm) == Array([[[[1, 0], [0, 0]], [[0, 1], [0, 0]]], [[[0, 0], [1, 0]], [[0, 0], [0, 1]]]])
    assert A.diff(a) == MutableDenseMatrix([[0, 0], [0, 0]])
    B = ImmutableDenseMatrix([a, b])
    assert A.diff(B) == Array.zeros(2, 1, 2, 2)
    assert A.diff(A) == Array([[[[1, 0], [0, 0]], [[0, 1], [0, 0]]], [[[0, 0], [1, 0]], [[0, 0], [0, 1]]]])
    dB = B.diff([[a, b]])
    assert dB.shape == (2, 2, 1)
    assert dB == Array([[[1], [0]], [[0], [1]]])
    f = Function('f')
    fxyz = f(x, y, z)
    assert fxyz.diff([[x, y, z]]) == Array([fxyz.diff(x), fxyz.diff(y), fxyz.diff(z)])
    assert fxyz.diff(([x, y, z], 2)) == Array([[fxyz.diff(x, 2), fxyz.diff(x, y), fxyz.diff(x, z)], [fxyz.diff(x, y), fxyz.diff(y, 2), fxyz.diff(y, z)], [fxyz.diff(x, z), fxyz.diff(z, y), fxyz.diff(z, 2)]])
    expr = sin(x) * exp(y)
    assert expr.diff([[x, y]]) == Array([cos(x) * exp(y), sin(x) * exp(y)])
    assert expr.diff(y, ((x, y),)) == Array([cos(x) * exp(y), sin(x) * exp(y)])
    assert expr.diff(x, ((x, y),)) == Array([-sin(x) * exp(y), cos(x) * exp(y)])
    assert expr.diff(((y, x),), [[x, y]]) == Array([[cos(x) * exp(y), -sin(x) * exp(y)], [sin(x) * exp(y), cos(x) * exp(y)]])
    assert fxyz.diff(x).diff(y).diff(x) == fxyz.diff(((x, y, z),), 3)[0, 1, 0]
    assert fxyz.diff(z).diff(y).diff(x) == fxyz.diff(((x, y, z),), 3)[2, 1, 0]
    assert fxyz.diff([[x, y, z]], ((z, y, x),)) == Array([[fxyz.diff(i).diff(j) for i in (x, y, z)] for j in (z, y, x)])
    res = x.diff(Matrix([[x, y]]))
    assert isinstance(res, ImmutableDenseMatrix)
    assert res == Matrix([[1, 0]])
    res = (x ** 3).diff(Matrix([[x, y]]))
    assert isinstance(res, ImmutableDenseMatrix)
    assert res == Matrix([[3 * x ** 2, 0]])