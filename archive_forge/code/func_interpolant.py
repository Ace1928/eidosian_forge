from bisect import bisect
from ..libmp.backend import xrange
def interpolant(x):
    x = ctx.convert(x)
    orig = ctx.prec
    try:
        ctx.prec = workprec
        ser, xa, xb = get_series(x)
        y = mpolyval(ser, x - xa)
    finally:
        ctx.prec = orig
    if return_vector:
        return [+yk for yk in y]
    else:
        return +y[0]