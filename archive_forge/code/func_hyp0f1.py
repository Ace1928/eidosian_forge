from ..libmp.backend import xrange
from .functions import defun, defun_wrapped
@defun
def hyp0f1(ctx, b, z, **kwargs):
    return ctx.hyper([], [b], z, **kwargs)