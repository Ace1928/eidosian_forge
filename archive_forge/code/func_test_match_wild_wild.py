from sympy import abc
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.mul import Mul
from sympy.core.numbers import (Float, I, Integer, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, Wild, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.hyper import meijerg
from sympy.polys.polytools import Poly
from sympy.simplify.radsimp import collect
from sympy.simplify.simplify import signsimp
from sympy.testing.pytest import XFAIL
def test_match_wild_wild():
    p = Wild('p')
    q = Wild('q')
    r = Wild('r')
    assert p.match(q + r) in [{q: p, r: 0}, {q: 0, r: p}]
    assert p.match(q * r) in [{q: p, r: 1}, {q: 1, r: p}]
    p = Wild('p')
    q = Wild('q', exclude=[p])
    r = Wild('r')
    assert p.match(q + r) == {q: 0, r: p}
    assert p.match(q * r) == {q: 1, r: p}
    p = Wild('p')
    q = Wild('q', exclude=[p])
    r = Wild('r', exclude=[p])
    assert p.match(q + r) is None
    assert p.match(q * r) is None