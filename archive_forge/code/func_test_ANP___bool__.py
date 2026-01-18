from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.domains import ZZ, QQ
from sympy.polys.polyclasses import DMP, DMF, ANP
from sympy.polys.polyerrors import (CoercionFailed, ExactQuotientFailed,
from sympy.polys.specialpolys import f_polys
from sympy.testing.pytest import raises
def test_ANP___bool__():
    assert bool(ANP([], [QQ(1), QQ(0), QQ(1)], QQ)) is False
    assert bool(ANP([QQ(1)], [QQ(1), QQ(0), QQ(1)], QQ)) is True