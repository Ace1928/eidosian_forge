from sympy.polys.galoistools import (
from sympy.polys.polyerrors import (
from sympy.polys import polyconfig as config
from sympy.polys.domains import ZZ
from sympy.core.numbers import pi
from sympy.ntheory.generate import nextprime
from sympy.testing.pytest import raises
def test_gf_normal():
    assert gf_normal([11, 22, 17, 1, 0], 11, ZZ) == [6, 1, 0]