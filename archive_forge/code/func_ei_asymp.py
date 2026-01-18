import operator
import math
import cmath
def ei_asymp(z, _e1=False):
    r = 1.0 / z
    s = t = 1.0
    k = 1
    while 1:
        t *= k * r
        s += t
        if abs(t) < 1e-16:
            break
        k += 1
    v = s * exp(z) / z
    if _e1:
        if type(z) is complex:
            zreal = z.real
            zimag = z.imag
        else:
            zreal = z
            zimag = 0.0
        if zimag == 0.0 and zreal > 0.0:
            v += pi * 1j
    elif type(z) is complex:
        if z.imag > 0:
            v += pi * 1j
        if z.imag < 0:
            v -= pi * 1j
    return v