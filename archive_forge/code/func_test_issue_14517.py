import random
from sympy.core.numbers import I
from sympy.core.numbers import Rational
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.polytools import Poly
from sympy.matrices import Matrix, eye, ones
from sympy.abc import x, y, z
from sympy.testing.pytest import raises
from sympy.matrices.common import NonSquareMatrixError
from sympy.functions.combinatorial.factorials import factorial, subfactorial
def test_issue_14517():
    M = Matrix([[0, 10 * I, 10 * I, 0], [10 * I, 0, 0, 10 * I], [10 * I, 0, 5 + 2 * I, 10 * I], [0, 10 * I, 10 * I, 5 + 2 * I]])
    ev = M.eigenvals()
    test_ev = random.choice(list(ev.keys()))
    assert (M - test_ev * eye(4)).det() == 0