from mpmath import mp
from mpmath import libmp
def test_levin_1():
    mp.dps = 17
    eps = mp.mpf(mp.eps)
    with mp.extraprec(2 * mp.prec):
        L = mp.levin(method='levin', variant='v')
        A, n = ([], 1)
        while 1:
            s = mp.mpf(n) ** (2 + 3j)
            n += 1
            A.append(s)
            v, e = L.update(A)
            if e < eps:
                break
            if n > 1000:
                raise RuntimeError('iteration limit exceeded')
    eps = mp.exp(0.9 * mp.log(eps))
    err = abs(v - mp.zeta(-2 - 3j))
    assert err < eps
    w = mp.nsum(lambda n: n ** (2 + 3j), [1, mp.inf], method='levin', levin_variant='v')
    err = abs(v - w)
    assert err < eps