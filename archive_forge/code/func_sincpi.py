from ..libmp.backend import xrange
import math
import cmath
@defun_wrapped
def sincpi(ctx, x):
    if ctx.isinf(x):
        return 1 / x
    if not x:
        return x + 1
    return ctx.sinpi(x) / (ctx.pi * x)