from sympy.core.mul import Mul
from sympy.core.numbers import (Integer, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.integrals.integrals import Integral
from sympy.testing.pytest import raises
from sympy.polys.polyutils import (
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.domains import ZZ
def test__sort_factors():
    assert _sort_factors([], multiple=True) == []
    assert _sort_factors([], multiple=False) == []
    F = [[1, 2, 3], [1, 2], [1]]
    G = [[1], [1, 2], [1, 2, 3]]
    assert _sort_factors(F, multiple=False) == G
    F = [[1, 2], [1, 2, 3], [1, 2], [1]]
    G = [[1], [1, 2], [1, 2], [1, 2, 3]]
    assert _sort_factors(F, multiple=False) == G
    F = [[2, 2], [1, 2, 3], [1, 2], [1]]
    G = [[1], [1, 2], [2, 2], [1, 2, 3]]
    assert _sort_factors(F, multiple=False) == G
    F = [([1, 2, 3], 1), ([1, 2], 1), ([1], 1)]
    G = [([1], 1), ([1, 2], 1), ([1, 2, 3], 1)]
    assert _sort_factors(F, multiple=True) == G
    F = [([1, 2], 1), ([1, 2, 3], 1), ([1, 2], 1), ([1], 1)]
    G = [([1], 1), ([1, 2], 1), ([1, 2], 1), ([1, 2, 3], 1)]
    assert _sort_factors(F, multiple=True) == G
    F = [([2, 2], 1), ([1, 2, 3], 1), ([1, 2], 1), ([1], 1)]
    G = [([1], 1), ([1, 2], 1), ([2, 2], 1), ([1, 2, 3], 1)]
    assert _sort_factors(F, multiple=True) == G
    F = [([2, 2], 1), ([1, 2, 3], 1), ([1, 2], 2), ([1], 1)]
    G = [([1], 1), ([2, 2], 1), ([1, 2], 2), ([1, 2, 3], 1)]
    assert _sort_factors(F, multiple=True) == G