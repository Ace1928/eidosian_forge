from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.matrices.dense import Matrix
from sympy.ntheory.factor_ import factorint
from sympy.simplify.powsimp import powsimp
from sympy.core.function import _mexpand
from sympy.core.sorting import default_sort_key, ordered
from sympy.functions.elementary.trigonometric import sin
from sympy.solvers.diophantine import diophantine
from sympy.solvers.diophantine.diophantine import (diop_DN,
from sympy.testing.pytest import slow, raises, XFAIL
from sympy.utilities.iterables import (
def test_holzer():
    assert holzer(2, 7, 13, 4, 79, 23) == (2, 7, 13)
    assert holzer(2, 6, 2, 1, 1, 10) == (2, 6, 2)
    raises(ValueError, lambda: holzer(2, 7, 14, 4, 79, 23))