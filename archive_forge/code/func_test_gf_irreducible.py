from sympy.polys.galoistools import (
from sympy.polys.polyerrors import (
from sympy.polys import polyconfig as config
from sympy.polys.domains import ZZ
from sympy.core.numbers import pi
from sympy.ntheory.generate import nextprime
from sympy.testing.pytest import raises
def test_gf_irreducible():
    assert gf_irreducible_p(gf_irreducible(1, 11, ZZ), 11, ZZ) is True
    assert gf_irreducible_p(gf_irreducible(2, 11, ZZ), 11, ZZ) is True
    assert gf_irreducible_p(gf_irreducible(3, 11, ZZ), 11, ZZ) is True
    assert gf_irreducible_p(gf_irreducible(4, 11, ZZ), 11, ZZ) is True
    assert gf_irreducible_p(gf_irreducible(5, 11, ZZ), 11, ZZ) is True
    assert gf_irreducible_p(gf_irreducible(6, 11, ZZ), 11, ZZ) is True
    assert gf_irreducible_p(gf_irreducible(7, 11, ZZ), 11, ZZ) is True