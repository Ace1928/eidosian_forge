from ..libmp.backend import xrange
import math
import cmath
@defun
def stirling1(ctx, n, k, exact=False):
    v = ctx._stirling1(int(n), int(k))
    if exact:
        return int(v)
    else:
        return ctx.mpf(v)