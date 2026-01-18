from ..libmp.backend import xrange
from .functions import defun, defun_wrapped
@defun
def hyp1f1(ctx, a, b, z, **kwargs):
    return ctx.hyper([a], [b], z, **kwargs)