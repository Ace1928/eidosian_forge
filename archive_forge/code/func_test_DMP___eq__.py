from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.domains import ZZ, QQ
from sympy.polys.polyclasses import DMP, DMF, ANP
from sympy.polys.polyerrors import (CoercionFailed, ExactQuotientFailed,
from sympy.polys.specialpolys import f_polys
from sympy.testing.pytest import raises
def test_DMP___eq__():
    assert DMP([[ZZ(1), ZZ(2)], [ZZ(3)]], ZZ) == DMP([[ZZ(1), ZZ(2)], [ZZ(3)]], ZZ)
    assert DMP([[ZZ(1), ZZ(2)], [ZZ(3)]], ZZ) == DMP([[QQ(1), QQ(2)], [QQ(3)]], QQ)
    assert DMP([[QQ(1), QQ(2)], [QQ(3)]], QQ) == DMP([[ZZ(1), ZZ(2)], [ZZ(3)]], ZZ)
    assert DMP([[[ZZ(1)]]], ZZ) != DMP([[ZZ(1)]], ZZ)
    assert DMP([[ZZ(1)]], ZZ) != DMP([[[ZZ(1)]]], ZZ)