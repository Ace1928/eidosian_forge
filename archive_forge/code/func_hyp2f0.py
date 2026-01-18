from ..libmp.backend import xrange
from .functions import defun, defun_wrapped
@defun
def hyp2f0(ctx, a, b, z, **kwargs):
    return ctx.hyper([a, b], [], z, **kwargs)