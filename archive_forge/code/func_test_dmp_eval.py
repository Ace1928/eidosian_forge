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
def test_dmp_eval():
    assert dmp_eval([], 3, 0, ZZ) == 0
    assert dmp_eval([[]], 3, 1, ZZ) == []
    assert dmp_eval([[[]]], 3, 2, ZZ) == [[]]
    assert dmp_eval([[1, 2]], 0, 1, ZZ) == [1, 2]
    assert dmp_eval([[[1]]], 3, 2, ZZ) == [[1]]
    assert dmp_eval([[[1, 2]]], 3, 2, ZZ) == [[1, 2]]
    assert dmp_eval([[3, 2], [1, 2]], 3, 1, ZZ) == [10, 8]
    assert dmp_eval([[[3, 2]], [[1, 2]]], 3, 2, ZZ) == [[10, 8]]