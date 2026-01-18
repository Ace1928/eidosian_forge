from sympy.polys.galoistools import (
from sympy.polys.polyerrors import (
from sympy.polys import polyconfig as config
from sympy.polys.domains import ZZ
from sympy.core.numbers import pi
from sympy.ntheory.generate import nextprime
from sympy.testing.pytest import raises
def test_gf_compose():
    assert gf_compose([], [1, 0], 11, ZZ) == []
    assert gf_compose_mod([], [1, 0], [1, 0], 11, ZZ) == []
    assert gf_compose([1], [], 11, ZZ) == [1]
    assert gf_compose([1, 0], [], 11, ZZ) == []
    assert gf_compose([1, 0], [1, 0], 11, ZZ) == [1, 0]
    f = ZZ.map([1, 1, 4, 9, 1])
    g = ZZ.map([1, 1, 1])
    h = ZZ.map([1, 0, 0, 2])
    assert gf_compose(g, h, 11, ZZ) == [1, 0, 0, 5, 0, 0, 7]
    assert gf_compose_mod(g, h, f, 11, ZZ) == [3, 9, 6, 10]