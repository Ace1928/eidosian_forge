from sympy.polys.densebasic import (
from sympy.polys.densearith import dmp_mul_ground
from sympy.polys.densetools import (
from sympy.polys.polyclasses import ANP
from sympy.polys.polyerrors import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import FF, ZZ, QQ, EX
from sympy.polys.rings import ring
from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.functions.elementary.trigonometric import sin
from sympy.abc import x
from sympy.testing.pytest import raises
def test_dmp_integrate_in():
    f = dmp_convert(f_6, 3, ZZ, QQ)
    assert dmp_integrate_in(f, 2, 1, 3, QQ) == dmp_swap(dmp_integrate(dmp_swap(f, 0, 1, 3, QQ), 2, 3, QQ), 0, 1, 3, QQ)
    assert dmp_integrate_in(f, 3, 1, 3, QQ) == dmp_swap(dmp_integrate(dmp_swap(f, 0, 1, 3, QQ), 3, 3, QQ), 0, 1, 3, QQ)
    assert dmp_integrate_in(f, 2, 2, 3, QQ) == dmp_swap(dmp_integrate(dmp_swap(f, 0, 2, 3, QQ), 2, 3, QQ), 0, 2, 3, QQ)
    assert dmp_integrate_in(f, 3, 2, 3, QQ) == dmp_swap(dmp_integrate(dmp_swap(f, 0, 2, 3, QQ), 3, 3, QQ), 0, 2, 3, QQ)