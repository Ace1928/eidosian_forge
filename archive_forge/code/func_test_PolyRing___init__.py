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
def test_PolyRing___init__():
    x, y, z, t = map(Symbol, 'xyzt')
    assert len(PolyRing('x,y,z', ZZ, lex).gens) == 3
    assert len(PolyRing(x, ZZ, lex).gens) == 1
    assert len(PolyRing(('x', 'y', 'z'), ZZ, lex).gens) == 3
    assert len(PolyRing((x, y, z), ZZ, lex).gens) == 3
    assert len(PolyRing('', ZZ, lex).gens) == 0
    assert len(PolyRing([], ZZ, lex).gens) == 0
    raises(GeneratorsError, lambda: PolyRing(0, ZZ, lex))
    assert PolyRing('x', ZZ[t], lex).domain == ZZ[t]
    assert PolyRing('x', 'ZZ[t]', lex).domain == ZZ[t]
    assert PolyRing('x', PolyRing('t', ZZ, lex), lex).domain == ZZ[t]
    raises(GeneratorsError, lambda: PolyRing('x', PolyRing('x', ZZ, lex), lex))
    _lex = Symbol('lex')
    assert PolyRing('x', ZZ, lex).order == lex
    assert PolyRing('x', ZZ, _lex).order == lex
    assert PolyRing('x', ZZ, 'lex').order == lex
    R1 = PolyRing('x,y', ZZ, lex)
    R2 = PolyRing('x,y', ZZ, lex)
    R3 = PolyRing('x,y,z', ZZ, lex)
    assert R1.x == R1.gens[0]
    assert R1.y == R1.gens[1]
    assert R1.x == R2.x
    assert R1.y == R2.y
    assert R1.x != R3.x
    assert R1.y != R3.y