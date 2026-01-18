from sympy.polys.galoistools import (
from sympy.polys.polyerrors import (
from sympy.polys import polyconfig as config
from sympy.polys.domains import ZZ
from sympy.core.numbers import pi
from sympy.ntheory.generate import nextprime
from sympy.testing.pytest import raises
def test_gf_cofactors():
    assert gf_cofactors(ZZ.map([]), ZZ.map([]), 11, ZZ) == ([], [], [])
    assert gf_cofactors(ZZ.map([2]), ZZ.map([]), 11, ZZ) == ([1], [2], [])
    assert gf_cofactors(ZZ.map([]), ZZ.map([2]), 11, ZZ) == ([1], [], [2])
    assert gf_cofactors(ZZ.map([2]), ZZ.map([2]), 11, ZZ) == ([1], [2], [2])
    assert gf_cofactors(ZZ.map([]), ZZ.map([1, 0]), 11, ZZ) == ([1, 0], [], [1])
    assert gf_cofactors(ZZ.map([1, 0]), ZZ.map([]), 11, ZZ) == ([1, 0], [1], [])
    assert gf_cofactors(ZZ.map([3, 0]), ZZ.map([3, 0]), 11, ZZ) == ([1, 0], [3], [3])
    assert gf_cofactors(ZZ.map([1, 8, 7]), ZZ.map([1, 7, 1, 7]), 11, ZZ) == ([1, 7], [1, 1], [1, 0, 1])