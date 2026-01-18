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
def test_dmp_lift():
    q = [QQ(1, 1), QQ(0, 1), QQ(1, 1)]
    f = [ANP([QQ(1, 1)], q, QQ), ANP([], q, QQ), ANP([], q, QQ), ANP([QQ(1, 1), QQ(0, 1)], q, QQ), ANP([QQ(17, 1), QQ(0, 1)], q, QQ)]
    assert dmp_lift(f, 0, QQ.algebraic_field(I)) == [QQ(1), QQ(0), QQ(0), QQ(0), QQ(0), QQ(0), QQ(2), QQ(0), QQ(578), QQ(0), QQ(0), QQ(0), QQ(1), QQ(0), QQ(-578), QQ(0), QQ(83521)]
    raises(DomainError, lambda: dmp_lift([EX(1), EX(2)], 0, EX))