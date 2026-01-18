from sympy.polys.densebasic import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import ZZ, QQ
from sympy.polys.rings import ring
from sympy.core.singleton import S
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_dup_reverse():
    assert dup_reverse([1, 2, 0, 3]) == [3, 0, 2, 1]
    assert dup_reverse([1, 2, 3, 0]) == [3, 2, 1]