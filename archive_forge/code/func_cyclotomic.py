from ..libmp.backend import xrange
import math
import cmath
@defun_wrapped
def cyclotomic(ctx, n, z):
    n = int(n)
    if n < 0:
        raise ValueError('n cannot be negative')
    p = ctx.one
    if n == 0:
        return p
    if n == 1:
        return z - p
    if n == 2:
        return z + p
    a_prod = 1
    b_prod = 1
    num_zeros = 0
    num_poles = 0
    for d in range(1, n + 1):
        if not n % d:
            w = ctx.moebius(n // d)
            b = -ctx.powm1(z, d)
            if b:
                p *= b ** w
            elif w == 1:
                a_prod *= d
                num_zeros += 1
            elif w == -1:
                b_prod *= d
                num_poles += 1
    if num_zeros:
        if num_zeros > num_poles:
            p *= 0
        else:
            p *= a_prod
            p /= b_prod
    return p