from mpmath import mp
from mpmath import libmp
def test_cohen_alt_0():
    mp.dps = 17
    AC = mp.cohen_alt()
    S, s, n = ([], 0, 1)
    while 1:
        s += -(-1) ** n * mp.one / (n * n)
        n += 1
        S.append(s)
        v, e = AC.update_psum(S)
        if e < mp.eps:
            break
        if n > 1000:
            raise RuntimeError('iteration limit exceeded')
    eps = mp.exp(0.9 * mp.log(mp.eps))
    err = abs(v - mp.pi ** 2 / 12)
    assert err < eps