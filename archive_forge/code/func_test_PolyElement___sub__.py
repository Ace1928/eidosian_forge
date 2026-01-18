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
def test_PolyElement___sub__():
    Rt, t = ring('t', ZZ)
    Ruv, u, v = ring('u,v', ZZ)
    Rxyz, x, y, z = ring('x,y,z', Ruv)
    assert dict(x - 3 * y) == {(1, 0, 0): 1, (0, 1, 0): -3}
    assert dict(-u + x) == dict(x - u) == {(1, 0, 0): 1, (0, 0, 0): -u}
    assert dict(-u + x * y) == dict(x * y - u) == {(1, 1, 0): 1, (0, 0, 0): -u}
    assert dict(-u + x * y + z) == dict(x * y + z - u) == {(1, 1, 0): 1, (0, 0, 1): 1, (0, 0, 0): -u}
    assert dict(-u * x + x) == dict(x - u * x) == {(1, 0, 0): -u + 1}
    assert dict(-u * x + x * y) == dict(x * y - u * x) == {(1, 1, 0): 1, (1, 0, 0): -u}
    assert dict(-u * x + x * y + z) == dict(x * y + z - u * x) == {(1, 1, 0): 1, (0, 0, 1): 1, (1, 0, 0): -u}
    raises(TypeError, lambda: t - x)
    raises(TypeError, lambda: x - t)
    raises(TypeError, lambda: t - u)
    raises(TypeError, lambda: u - t)
    Fuv, u, v = field('u,v', ZZ)
    Rxyz, x, y, z = ring('x,y,z', Fuv)
    assert dict(-u + x) == dict(x - u) == {(1, 0, 0): 1, (0, 0, 0): -u}
    Rxyz, x, y, z = ring('x,y,z', EX)
    assert dict(-EX(pi) + x * y * z) == dict(x * y * z - EX(pi)) == {(1, 1, 1): EX(1), (0, 0, 0): -EX(pi)}