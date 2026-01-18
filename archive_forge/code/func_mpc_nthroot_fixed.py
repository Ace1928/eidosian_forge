import sys
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, BACKEND
from .libmpf import (\
from .libelefun import (\
def mpc_nthroot_fixed(a, b, n, prec):
    start = 50
    a1 = int(rshift(a, prec - n * start))
    b1 = int(rshift(b, prec - n * start))
    try:
        r = (a1 + 1j * b1) ** (1.0 / n)
        re = r.real
        im = r.imag
        re = MPZ(int(re))
        im = MPZ(int(im))
    except OverflowError:
        a1 = from_int(a1, start)
        b1 = from_int(b1, start)
        fn = from_int(n)
        nth = mpf_rdiv_int(1, fn, start)
        re, im = mpc_pow((a1, b1), (nth, fzero), start)
        re = to_int(re)
        im = to_int(im)
    extra = 10
    prevp = start
    extra1 = n
    for p in giant_steps(start, prec + extra):
        re2, im2 = complex_int_pow(re, im, n - 1)
        re2 = rshift(re2, (n - 1) * prevp - p - extra1)
        im2 = rshift(im2, (n - 1) * prevp - p - extra1)
        r4 = re2 * re2 + im2 * im2 >> p + extra1
        ap = rshift(a, prec - p)
        bp = rshift(b, prec - p)
        rec = ap * re2 + bp * im2 >> p
        imc = -ap * im2 + bp * re2 >> p
        reb = (rec << p) // r4
        imb = (imc << p) // r4
        re = (reb + (n - 1) * lshift(re, p - prevp)) // n
        im = (imb + (n - 1) * lshift(im, p - prevp)) // n
        prevp = p
    return (re, im)