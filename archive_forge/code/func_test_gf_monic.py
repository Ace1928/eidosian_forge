from sympy.polys.galoistools import (
from sympy.polys.polyerrors import (
from sympy.polys import polyconfig as config
from sympy.polys.domains import ZZ
from sympy.core.numbers import pi
from sympy.ntheory.generate import nextprime
from sympy.testing.pytest import raises
def test_gf_monic():
    assert gf_monic(ZZ.map([]), 11, ZZ) == (0, [])
    assert gf_monic(ZZ.map([1]), 11, ZZ) == (1, [1])
    assert gf_monic(ZZ.map([2]), 11, ZZ) == (2, [1])
    assert gf_monic(ZZ.map([1, 2, 3, 4]), 11, ZZ) == (1, [1, 2, 3, 4])
    assert gf_monic(ZZ.map([2, 3, 4, 5]), 11, ZZ) == (2, [1, 7, 2, 8])