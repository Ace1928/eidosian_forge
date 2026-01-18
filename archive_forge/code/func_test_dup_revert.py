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
def test_dup_revert():
    f = [-QQ(1, 720), QQ(0), QQ(1, 24), QQ(0), -QQ(1, 2), QQ(0), QQ(1)]
    g = [QQ(61, 720), QQ(0), QQ(5, 24), QQ(0), QQ(1, 2), QQ(0), QQ(1)]
    assert dup_revert(f, 8, QQ) == g
    raises(NotReversible, lambda: dup_revert([QQ(1), QQ(0)], 3, QQ))