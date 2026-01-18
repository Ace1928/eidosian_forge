from mpmath import *
from mpmath.libmp import *
def test_float_rounding():
    mp.prec = 64
    for x in [mpf(1), mpf(1) + eps, mpf(1) - eps, -mpf(1) + eps, -mpf(1) - eps]:
        fa = float(x)
        fb = float(fadd(x, 0, prec=53, rounding='n'))
        assert fa == fb
        z = mpc(x, x)
        ca = complex(z)
        cb = complex(fadd(z, 0, prec=53, rounding='n'))
        assert ca == cb
        for rnd in ['n', 'd', 'u', 'f', 'c']:
            fa = to_float(x._mpf_, rnd=rnd)
            fb = to_float(fadd(x, 0, prec=53, rounding=rnd)._mpf_, rnd=rnd)
            assert fa == fb
    mp.prec = 53