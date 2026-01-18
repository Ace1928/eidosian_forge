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
def test_matrix_norm():
    x = Symbol('x', real=True)
    v = Matrix([cos(x), sin(x)])
    assert trigsimp(v.norm(2)) == 1
    assert v.norm(10) == Pow(cos(x) ** 10 + sin(x) ** 10, Rational(1, 10))
    A = Matrix([[5, Rational(3, 2)]])
    assert A.norm() == Pow(25 + Rational(9, 4), S.Half)
    assert A.norm(oo) == max(A)
    assert A.norm(-oo) == min(A)
    A = Matrix([[1, 1], [1, 1]])
    assert A.norm(2) == 2
    assert A.norm(-2) == 0
    assert A.norm('frobenius') == 2
    assert eye(10).norm(2) == eye(10).norm(-2) == 1
    assert A.norm(oo) == 2
    A = Matrix([[3, y, y], [x, S.Half, -pi]])
    assert A.norm('fro') == sqrt(Rational(37, 4) + 2 * abs(y) ** 2 + pi ** 2 + x ** 2)
    A = Matrix([[1, 2, -3], [4, 5, Rational(13, 2)]])
    assert A.norm(2) == sqrt(Rational(389, 8) + sqrt(78665) / 8)
    assert A.norm(-2) is S.Zero
    assert A.norm('frobenius') == sqrt(389) / 2
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 5], [-2, 2]])
    C = Matrix([[0, -I], [I, 0]])
    D = Matrix([[1, 0], [0, -1]])
    L = [A, B, C, D]
    alpha = Symbol('alpha', real=True)
    for order in ['fro', 2, -2]:
        assert zeros(3).norm(order) is S.Zero
        for X in L:
            for Y in L:
                dif = X.norm(order) + Y.norm(order) - (X + Y).norm(order)
                assert dif >= 0
        for M in [A, B, C, D]:
            dif = simplify((alpha * M).norm(order) - abs(alpha) * M.norm(order))
            assert dif == 0
    a = Matrix([1, 1 - 1 * I, -3])
    b = Matrix([S.Half, 1 * I, 1])
    c = Matrix([-1, -1, -1])
    d = Matrix([3, 2, I])
    e = Matrix([Integer(100.0), Rational(1, 100.0), 1])
    L = [a, b, c, d, e]
    alpha = Symbol('alpha', real=True)
    for order in [1, 2, -1, -2, S.Infinity, S.NegativeInfinity, pi]:
        if order > 0:
            assert Matrix([0, 0, 0]).norm(order) is S.Zero
        if order >= 1:
            for X in L:
                for Y in L:
                    dif = X.norm(order) + Y.norm(order) - (X + Y).norm(order)
                    assert simplify(dif >= 0) is S.true
        if order in [1, 2, -1, -2, S.Infinity, S.NegativeInfinity]:
            for X in L:
                dif = simplify((alpha * X).norm(order) - abs(alpha) * X.norm(order))
                assert dif == 0
    M = Matrix(3, 3, [1, 3, 0, -2, -1, 0, 3, 9, 6])
    assert M.norm(1) == 13