from sympy.polys.galoistools import (
from sympy.polys.polyerrors import (
from sympy.polys import polyconfig as config
from sympy.polys.domains import ZZ
from sympy.core.numbers import pi
from sympy.ntheory.generate import nextprime
from sympy.testing.pytest import raises
def test_gf_int():
    assert gf_int(0, 5) == 0
    assert gf_int(1, 5) == 1
    assert gf_int(2, 5) == 2
    assert gf_int(3, 5) == -2
    assert gf_int(4, 5) == -1
    assert gf_int(5, 5) == 0