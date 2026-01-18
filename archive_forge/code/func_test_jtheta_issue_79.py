import mpmath
import random
import pytest
from mpmath import *
def test_jtheta_issue_79():
    mp.dps = 30
    mp.dps += 30
    q = mpf(6) / 10 - one / 10 ** 6 - mpf(8) / 10 * j
    mp.dps -= 30
    res = mpf('32.0031009628901652627099524264') + mpf('16.6153027998236087899308935624') * j
    result = jtheta(3, 1, q)
    mp.dps += 30
    q = mpf(6) / 10 - one / 10 ** 7 - mpf(8) / 10 * j
    mp.dps -= 30
    pytest.raises(ValueError, lambda: jtheta(3, 1, q))
    mp.dps = 100
    z = (1 + j) / 3
    q = mpf(368983957219251) / 10 ** 15 + mpf(636363636363636) / 10 ** 15 * j
    res = mpf('2.4439389177990737589761828991467471') + mpf('0.5446453005688226915290954851851490') * j
    mp.dps = 30
    result = jtheta(1, z, q)
    assert result.ae(res)
    mp.dps = 80
    z = 3 + 4 * j
    q = 0.5 + 0.5 * j
    r1 = jtheta(1, z, q)
    mp.dps = 15
    r2 = jtheta(1, z, q)
    assert r1.ae(r2)
    mp.dps = 80
    z = 3 + j
    q1 = exp(j * 3)
    for n in range(1, 2):
        mp.dps = 80
        q = q1 * (1 - mpf(1) / 10 ** n)
        r1 = jtheta(1, z, q)
        mp.dps = 15
        r2 = jtheta(1, z, q)
    assert r1.ae(r2)
    mp.dps = 15
    assert jtheta(3, 4.5, 0.25, 9).ae(1359.04892680683)
    assert jtheta(3, 4.5, 0.25, 50).ae(-6.14832772630905e+33)
    mp.dps = 50
    r = jtheta(3, 4.5, 0.25, 9)
    assert r.ae('1359.048926806828939547859396600218966947753213803')
    r = jtheta(3, 4.5, 0.25, 50)
    assert r.ae('-6148327726309051673317975084654262.4119215720343656')