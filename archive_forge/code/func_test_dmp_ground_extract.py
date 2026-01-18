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
def test_dmp_ground_extract():
    f = dmp_normal([[2930944], [], [2198208], [], [549552], [], [45796]], 1, ZZ)
    g = dmp_normal([[17585664], [], [8792832], [], [1099104], []], 1, ZZ)
    F = dmp_normal([[64], [], [48], [], [12], [], [1]], 1, ZZ)
    G = dmp_normal([[384], [], [192], [], [24], []], 1, ZZ)
    assert dmp_ground_extract(f, g, 1, ZZ) == (45796, F, G)