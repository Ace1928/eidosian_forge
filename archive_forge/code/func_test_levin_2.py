from mpmath import mp
from mpmath import libmp
def test_levin_2():
    mp.dps = 17
    z = mp.mpf(10)
    eps = mp.mpf(mp.eps)
    with mp.extraprec(2 * mp.prec):
        L = mp.levin(method='sidi', variant='t')
        n = 0
        while 1:
            s = (-1) ** n * mp.fac(n) * z ** (-n)
            v, e = L.step(s)
            n += 1
            if e < eps:
                break
            if n > 1000:
                raise RuntimeError('iteration limit exceeded')
    eps = mp.exp(0.9 * mp.log(eps))
    exact = mp.quad(lambda x: mp.exp(-x) / (1 + x / z), [0, mp.inf])
    err = abs(v - exact)
    assert err < eps
    w = mp.nsum(lambda n: (-1) ** n * mp.fac(n) * z ** (-n), [0, mp.inf], method='sidi', levin_variant='t')
    assert err < eps