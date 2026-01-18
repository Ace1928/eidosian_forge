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
def test_dup_zz_irreducible_p():
    R, x = ring('x', ZZ)
    assert R.dup_zz_irreducible_p(3 * x ** 4 + 2 * x ** 3 + 6 * x ** 2 + 8 * x + 7) is None
    assert R.dup_zz_irreducible_p(3 * x ** 4 + 2 * x ** 3 + 6 * x ** 2 + 8 * x + 4) is None
    assert R.dup_zz_irreducible_p(3 * x ** 4 + 2 * x ** 3 + 6 * x ** 2 + 8 * x + 10) is True
    assert R.dup_zz_irreducible_p(3 * x ** 4 + 2 * x ** 3 + 6 * x ** 2 + 8 * x + 14) is True