from sympy.polys.galoistools import (
from sympy.polys.polyerrors import (
from sympy.polys import polyconfig as config
from sympy.polys.domains import ZZ
from sympy.core.numbers import pi
from sympy.ntheory.generate import nextprime
from sympy.testing.pytest import raises
def test_gf_strip():
    assert gf_strip([]) == []
    assert gf_strip([0]) == []
    assert gf_strip([0, 0, 0]) == []
    assert gf_strip([1]) == [1]
    assert gf_strip([0, 1]) == [1]
    assert gf_strip([0, 0, 0, 1]) == [1]
    assert gf_strip([1, 2, 0]) == [1, 2, 0]
    assert gf_strip([0, 1, 2, 0]) == [1, 2, 0]
    assert gf_strip([0, 0, 0, 1, 2, 0]) == [1, 2, 0]