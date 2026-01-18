import mpmath
import random
import pytest
from mpmath import *
def test_jtheta_complex():
    mp.dps = 30
    z = mpf(1) / 4 + j / 8
    q = mpf(1) / 3 + j / 7
    res = mpf('0.31618034835986160705729105731678285') + mpf('0.07542013825835103435142515194358975') * j
    r = jtheta(1, z, q)
    assert mpc_ae(r, res)
    res = mpf('1.6530986428239765928634711417951828') + mpf('0.2015344864707197230526742145361455') * j
    r = jtheta(2, z, q)
    assert mpc_ae(r, res)
    res = mpf('1.6520564411784228184326012700348340') + mpf('0.1998129119671271328684690067401823') * j
    r = jtheta(3, z, q)
    assert mpc_ae(r, res)
    res = mpf('0.37619082382228348252047624089973824') - mpf('0.15623022130983652972686227200681074') * j
    r = jtheta(4, z, q)
    assert mpc_ae(r, res)
    mp.dos = 100
    z = mpf(1) / 4 + j / 8
    q = mpf(1) / 3 + j / 7
    mp.dps += 10
    a = [0, 0, jtheta(2, 0, q), jtheta(3, 0, q), jtheta(4, 0, q)]
    t = [0, jtheta(1, z, q), jtheta(2, z, q), jtheta(3, z, q), jtheta(4, z, q)]
    r = [(t[2] * a[4]) ** 2 - (t[4] * a[2]) ** 2 + (t[1] * a[3]) ** 2, (t[3] * a[4]) ** 2 - (t[4] * a[3]) ** 2 + (t[1] * a[2]) ** 2, (t[1] * a[4]) ** 2 - (t[3] * a[2]) ** 2 + (t[2] * a[3]) ** 2, (t[4] * a[4]) ** 2 - (t[3] * a[3]) ** 2 + (t[2] * a[2]) ** 2, a[2] ** 4 + a[4] ** 4 - a[3] ** 4]
    mp.dps -= 10
    for x in r:
        assert mpc_ae(x, mpc(0))
    mp.dps = 15