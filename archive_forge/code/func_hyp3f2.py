from ..libmp.backend import xrange
from .functions import defun, defun_wrapped
@defun
def hyp3f2(ctx, a1, a2, a3, b1, b2, z, **kwargs):
    return ctx.hyper([a1, a2, a3], [b1, b2], z, **kwargs)