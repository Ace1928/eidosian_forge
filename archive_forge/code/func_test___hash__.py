from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.domains import ZZ, QQ
from sympy.polys.polyclasses import DMP, DMF, ANP
from sympy.polys.polyerrors import (CoercionFailed, ExactQuotientFailed,
from sympy.polys.specialpolys import f_polys
from sympy.testing.pytest import raises
def test___hash__():
    assert DMP([[1, 2], [3]], ZZ) == DMP([[int(1), int(2)], [int(3)]], ZZ)
    assert hash(DMP([[1, 2], [3]], ZZ)) == hash(DMP([[int(1), int(2)], [int(3)]], ZZ))
    assert DMF(([[1, 2], [3]], [[1]]), ZZ) == DMF(([[int(1), int(2)], [int(3)]], [[int(1)]]), ZZ)
    assert hash(DMF(([[1, 2], [3]], [[1]]), ZZ)) == hash(DMF(([[int(1), int(2)], [int(3)]], [[int(1)]]), ZZ))
    assert ANP([1, 1], [1, 0, 1], ZZ) == ANP([int(1), int(1)], [int(1), int(0), int(1)], ZZ)
    assert hash(ANP([1, 1], [1, 0, 1], ZZ)) == hash(ANP([int(1), int(1)], [int(1), int(0), int(1)], ZZ))