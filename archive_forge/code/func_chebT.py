from ..libmp.backend import xrange
from .calculus import defun
def chebT(ctx, a=1, b=0):
    Tb = [1]
    yield Tb
    Ta = [b, a]
    while 1:
        yield Ta
        Tmp = [0] + [2 * a * t for t in Ta]
        for i, c in enumerate(Ta):
            Tmp[i] += 2 * b * c
        for i, c in enumerate(Tb):
            Tmp[i] -= c
        Ta, Tb = (Tmp, Ta)