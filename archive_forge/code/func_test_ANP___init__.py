from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.domains import ZZ, QQ
from sympy.polys.polyclasses import DMP, DMF, ANP
from sympy.polys.polyerrors import (CoercionFailed, ExactQuotientFailed,
from sympy.polys.specialpolys import f_polys
from sympy.testing.pytest import raises
def test_ANP___init__():
    rep = [QQ(1), QQ(1)]
    mod = [QQ(1), QQ(0), QQ(1)]
    f = ANP(rep, mod, QQ)
    assert f.rep == [QQ(1), QQ(1)]
    assert f.mod == [QQ(1), QQ(0), QQ(1)]
    assert f.dom == QQ
    rep = {1: QQ(1), 0: QQ(1)}
    mod = {2: QQ(1), 0: QQ(1)}
    f = ANP(rep, mod, QQ)
    assert f.rep == [QQ(1), QQ(1)]
    assert f.mod == [QQ(1), QQ(0), QQ(1)]
    assert f.dom == QQ
    f = ANP(1, mod, QQ)
    assert f.rep == [QQ(1)]
    assert f.mod == [QQ(1), QQ(0), QQ(1)]
    assert f.dom == QQ
    f = ANP([1, 0.5], mod, QQ)
    assert all((QQ.of_type(a) for a in f.rep))
    raises(CoercionFailed, lambda: ANP([sqrt(2)], mod, QQ))