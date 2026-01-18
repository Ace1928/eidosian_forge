from functools import reduce
from operator import add, mul
from sympy.polys.rings import ring, xring, sring, PolyRing, PolyElement
from sympy.polys.fields import field, FracField
from sympy.polys.domains import ZZ, QQ, RR, FF, EX
from sympy.polys.orderings import lex, grlex
from sympy.polys.polyerrors import GeneratorsError, \
from sympy.testing.pytest import raises
from sympy.core import Symbol, symbols
from sympy.core.singleton import S
from sympy.core.numbers import (oo, pi)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
def test_PolyElement___mul__():
    Rt, t = ring('t', ZZ)
    Ruv, u, v = ring('u,v', ZZ)
    Rxyz, x, y, z = ring('x,y,z', Ruv)
    assert dict(u * x) == dict(x * u) == {(1, 0, 0): u}
    assert dict(2 * u * x + z) == dict(x * 2 * u + z) == {(1, 0, 0): 2 * u, (0, 0, 1): 1}
    assert dict(u * 2 * x + z) == dict(2 * x * u + z) == {(1, 0, 0): 2 * u, (0, 0, 1): 1}
    assert dict(2 * u * x + z) == dict(x * 2 * u + z) == {(1, 0, 0): 2 * u, (0, 0, 1): 1}
    assert dict(u * x * 2 + z) == dict(x * u * 2 + z) == {(1, 0, 0): 2 * u, (0, 0, 1): 1}
    assert dict(2 * u * x * y + z) == dict(x * y * 2 * u + z) == {(1, 1, 0): 2 * u, (0, 0, 1): 1}
    assert dict(u * 2 * x * y + z) == dict(2 * x * y * u + z) == {(1, 1, 0): 2 * u, (0, 0, 1): 1}
    assert dict(2 * u * x * y + z) == dict(x * y * 2 * u + z) == {(1, 1, 0): 2 * u, (0, 0, 1): 1}
    assert dict(u * x * y * 2 + z) == dict(x * y * u * 2 + z) == {(1, 1, 0): 2 * u, (0, 0, 1): 1}
    assert dict(2 * u * y * x + z) == dict(y * x * 2 * u + z) == {(1, 1, 0): 2 * u, (0, 0, 1): 1}
    assert dict(u * 2 * y * x + z) == dict(2 * y * x * u + z) == {(1, 1, 0): 2 * u, (0, 0, 1): 1}
    assert dict(2 * u * y * x + z) == dict(y * x * 2 * u + z) == {(1, 1, 0): 2 * u, (0, 0, 1): 1}
    assert dict(u * y * x * 2 + z) == dict(y * x * u * 2 + z) == {(1, 1, 0): 2 * u, (0, 0, 1): 1}
    assert dict(3 * u * (x + y) + z) == dict((x + y) * 3 * u + z) == {(1, 0, 0): 3 * u, (0, 1, 0): 3 * u, (0, 0, 1): 1}
    raises(TypeError, lambda: t * x + z)
    raises(TypeError, lambda: x * t + z)
    raises(TypeError, lambda: t * u + z)
    raises(TypeError, lambda: u * t + z)
    Fuv, u, v = field('u,v', ZZ)
    Rxyz, x, y, z = ring('x,y,z', Fuv)
    assert dict(u * x) == dict(x * u) == {(1, 0, 0): u}
    Rxyz, x, y, z = ring('x,y,z', EX)
    assert dict(EX(pi) * x * y * z) == dict(x * y * z * EX(pi)) == {(1, 1, 1): EX(pi)}