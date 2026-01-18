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
def test_dmp_qq_i_factor():
    R, x, y = ring('x, y', QQ_I)
    i = QQ_I(0, 1)
    assert R.dmp_qq_i_factor(x ** 2 + 2 * y ** 2) == (QQ_I(1, 0), [(x ** 2 + 2 * y ** 2, 1)])
    assert R.dmp_qq_i_factor(x ** 2 + y ** 2) == (QQ_I(1, 0), [(x - i * y, 1), (x + i * y, 1)])
    assert R.dmp_qq_i_factor(x ** 2 + y ** 2 / 4) == (QQ_I(1, 0), [(x - i * y / 2, 1), (x + i * y / 2, 1)])
    assert R.dmp_qq_i_factor(4 * x ** 2 + y ** 2) == (QQ_I(4, 0), [(x - i * y / 2, 1), (x + i * y / 2, 1)])