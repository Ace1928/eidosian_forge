from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.domains import ZZ, QQ
from sympy.polys.polyclasses import DMP, DMF, ANP
from sympy.polys.polyerrors import (CoercionFailed, ExactQuotientFailed,
from sympy.polys.specialpolys import f_polys
from sympy.testing.pytest import raises
def test_DMP_exclude():
    f = [[[[[[[[[[[[[[[[[[[[[[[[[[1]], [[]]]]]]]]]]]]]]]]]]]]]]]]]]
    J = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25]
    assert DMP(f, ZZ).exclude() == (J, DMP([1, 0], ZZ))
    assert DMP([[1], [1, 0]], ZZ).exclude() == ([], DMP([[1], [1, 0]], ZZ))