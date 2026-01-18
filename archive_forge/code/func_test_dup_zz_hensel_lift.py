from sympy.polys.rings import ring, xring
from sympy.polys.domains import FF, ZZ, QQ, ZZ_I, QQ_I, RR, EX
from sympy.polys import polyconfig as config
from sympy.polys.polyerrors import DomainError
from sympy.polys.polyclasses import ANP
from sympy.polys.specialpolys import f_polys, w_polys
from sympy.core.numbers import I
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.ntheory.generate import nextprime
from sympy.testing.pytest import raises, XFAIL
def test_dup_zz_hensel_lift():
    R, x = ring('x', ZZ)
    f = x ** 4 - 1
    F = [x - 1, x - 2, x + 2, x + 1]
    assert R.dup_zz_hensel_lift(ZZ(5), f, F, 4) == [x - 1, x - 182, x + 182, x + 1]