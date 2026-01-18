from ..libmp.backend import xrange
import math
import cmath
@defun_wrapped
def sec(ctx, z):
    return ctx.one / ctx.cos(z)