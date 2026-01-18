from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, AlgebraicField
from sympy.polys.modulargcd import (
from sympy.functions.elementary.miscellaneous import sqrt
def test_modgcd_func_field():
    D, t = ring('t', ZZ)
    R, x, z = ring('x, z', D)
    minpoly = (z ** 2 * t ** 2 + z ** 2 * t - 1).drop(0)
    f, g = (x + 1, x - 1)
    assert _func_field_modgcd_m(f, g, minpoly) == R.one