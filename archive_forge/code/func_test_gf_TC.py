from sympy.polys.galoistools import (
from sympy.polys.polyerrors import (
from sympy.polys import polyconfig as config
from sympy.polys.domains import ZZ
from sympy.core.numbers import pi
from sympy.ntheory.generate import nextprime
from sympy.testing.pytest import raises
def test_gf_TC():
    assert gf_TC([], ZZ) == 0
    assert gf_TC([1], ZZ) == 1
    assert gf_TC([1, 2], ZZ) == 2