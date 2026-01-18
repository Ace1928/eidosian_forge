from ..libmp.backend import xrange
import math
import cmath
@defun_wrapped
def asec(ctx, z):
    return ctx.acos(ctx.one / z)