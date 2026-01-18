from ..libmp.backend import xrange
import math
import cmath
@defun_wrapped
def powm1(ctx, x, y):
    mag = ctx.mag
    one = ctx.one
    w = x ** y - one
    M = mag(w)
    if M > -8:
        return w
    if not w:
        if not y or (x in (1, -1, 1j, -1j) and ctx.isint(y)):
            return w
    x1 = x - one
    magy = mag(y)
    lnx = ctx.ln(x)
    if magy + mag(lnx) < -ctx.prec:
        return lnx * y + (lnx * y) ** 2 / 2
    return ctx.sum_accurately(lambda: iter([x ** y, -1]), 1)