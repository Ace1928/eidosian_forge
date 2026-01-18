from mpmath import mp
from mpmath import libmp
def test_levin_0():
    mp.dps = 17
    eps = mp.mpf(mp.eps)
    with mp.extraprec(2 * mp.prec):
        L = mp.levin(method='levin', variant='u')
        S, s, n = ([], 0, 1)
        while 1:
            s += mp.one / (n * n)
            n += 1
            S.append(s)
            v, e = L.update_psum(S)
            if e < eps:
                break
            if n > 1000:
                raise RuntimeError('iteration limit exceeded')
    eps = mp.exp(0.9 * mp.log(eps))
    err = abs(v - mp.pi ** 2 / 6)
    assert err < eps
    w = mp.nsum(lambda n: 1 / (n * n), [1, mp.inf], method='levin', levin_variant='u')
    err = abs(v - w)
    assert err < eps