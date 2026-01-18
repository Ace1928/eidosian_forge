from sympy.polys.densebasic import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import ZZ, QQ
from sympy.polys.rings import ring
from sympy.core.singleton import S
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_dmp_from_to_dict():
    assert dmp_from_dict({}, 1, ZZ) == [[]]
    assert dmp_to_dict([[]], 1) == {}
    assert dmp_to_dict([], 0, ZZ, zero=True) == {(0,): ZZ(0)}
    assert dmp_to_dict([[]], 1, ZZ, zero=True) == {(0, 0): ZZ(0)}
    f = [[3], [], [], [2], [], [], [], [], [8]]
    g = {(8, 0): 3, (5, 0): 2, (0, 0): 8}
    assert dmp_from_dict(g, 1, ZZ) == f
    assert dmp_to_dict(f, 1) == g