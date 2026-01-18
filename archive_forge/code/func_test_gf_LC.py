from sympy.polys.galoistools import (
from sympy.polys.polyerrors import (
from sympy.polys import polyconfig as config
from sympy.polys.domains import ZZ
from sympy.core.numbers import pi
from sympy.ntheory.generate import nextprime
from sympy.testing.pytest import raises
def test_gf_LC():
    assert gf_LC([], ZZ) == 0
    assert gf_LC([1], ZZ) == 1
    assert gf_LC([1, 2], ZZ) == 1