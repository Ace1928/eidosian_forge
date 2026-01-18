import mpmath
import random
import pytest
from mpmath import *
def test_jtheta():
    mp.dps = 25
    z = q = zero
    for n in range(1, 5):
        value = jtheta(n, z, q)
        assert value == (n - 1) // 2
    for q in [one, mpf(2)]:
        for n in range(1, 5):
            pytest.raises(ValueError, lambda: jtheta(n, z, q))
    z = one / 10
    q = one / 11
    res = mpf('0.1069552990104042681962096')
    result = jtheta(1, z, q)
    assert result.ae(res)
    res = mpf('1.101385760258855791140606')
    result = jtheta(2, z, q)
    assert result.ae(res)
    res = mpf('1.178319743354331061795905')
    result = jtheta(3, z, q)
    assert result.ae(res)
    res = mpf('0.8219318954665153577314573')
    result = jtheta(4, z, q)
    assert result.ae(res)
    z1 = pi
    z2 = pi / 2
    for i in range(10):
        qstring = str(random.random())
        q = mpf(qstring)
        result = jtheta(1, z1, q)
        assert result.ae(0)
        result = jtheta(2, z2, q)
        assert result.ae(0)
    mp.dps = 15