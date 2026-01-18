import mpmath
import random
import pytest
from mpmath import *
def test_djtheta():
    mp.dps = 30
    z = one / 7 + j / 3
    q = one / 8 + j / 5
    res = mpf('1.5555195883277196036090928995803201') - mpf('0.02439761276895463494054149673076275') * j
    result = jtheta(1, z, q, 1)
    assert mpc_ae(result, res)
    res = mpf('0.19825296689470982332701283509685662') - mpf('0.46038135182282106983251742935250009') * j
    result = jtheta(2, z, q, 1)
    assert mpc_ae(result, res)
    res = mpf('0.36492498415476212680896699407390026') - mpf('0.57743812698666990209897034525640369') * j
    result = jtheta(3, z, q, 1)
    assert mpc_ae(result, res)
    res = mpf('-0.38936892528126996010818803742007352') + mpf('0.66549886179739128256269617407313625') * j
    result = jtheta(4, z, q, 1)
    assert mpc_ae(result, res)
    for i in range(10):
        q = (one * random.random() + j * random.random()) / 2
        a = jtheta(1, 0, q, 1)
        b = jtheta(2, 0, q) * jtheta(3, 0, q) * jtheta(4, 0, q)
        assert a.ae(b)
    mp.dps = 20
    for q, z in [(one / 3, one / 5), (one / 3 + j / 8, one / 5), (one / 3, one / 5 + j / 8), (one / 3 + j / 7, one / 5 + j / 8)]:
        for n in [1, 2, 3, 4]:
            r = jtheta(n, z, q, 2)
            r1 = diff(lambda zz: jtheta(n, zz, q), z, n=2)
            assert r.ae(r1)
            r = jtheta(n, z, q, 3)
            r1 = diff(lambda zz: jtheta(n, zz, q), z, n=3)
            assert r.ae(r1)
    q = one / 3
    z = zero
    a = [0] * 5
    a[1] = jtheta(1, z, q, 3) / jtheta(1, z, q, 1)
    for n in [2, 3, 4]:
        a[n] = jtheta(n, z, q, 2) / jtheta(n, z, q)
    equality = a[2] + a[3] + a[4] - a[1]
    assert equality.ae(0)
    mp.dps = 15