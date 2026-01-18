from ..libmp.backend import xrange
import math
import cmath
@defun_wrapped
def csch(ctx, z):
    return ctx.one / ctx.sinh(z)