from ..libmp.backend import xrange
import math
import cmath
@defun_wrapped
def coth(ctx, z):
    return ctx.one / ctx.tanh(z)