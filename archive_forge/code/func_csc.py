from ..libmp.backend import xrange
import math
import cmath
@defun_wrapped
def csc(ctx, z):
    return ctx.one / ctx.sin(z)