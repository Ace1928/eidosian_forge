from sympy.polys.galoistools import (
from sympy.polys.polyerrors import (
from sympy.polys import polyconfig as config
from sympy.polys.domains import ZZ
from sympy.core.numbers import pi
from sympy.ntheory.generate import nextprime
from sympy.testing.pytest import raises
def test_gf_shift():
    f = [1, 2, 3, 4, 5]
    assert gf_lshift([], 5, ZZ) == []
    assert gf_rshift([], 5, ZZ) == ([], [])
    assert gf_lshift(f, 1, ZZ) == [1, 2, 3, 4, 5, 0]
    assert gf_lshift(f, 2, ZZ) == [1, 2, 3, 4, 5, 0, 0]
    assert gf_rshift(f, 0, ZZ) == (f, [])
    assert gf_rshift(f, 1, ZZ) == ([1, 2, 3, 4], [5])
    assert gf_rshift(f, 3, ZZ) == ([1, 2], [3, 4, 5])
    assert gf_rshift(f, 5, ZZ) == ([], f)