from ..libmp.backend import xrange
from .functions import defun, defun_wrapped
@defun
def appellf2(ctx, a, b1, b2, c1, c2, x, y, **kwargs):
    return ctx.hyper2d({'m+n': [a], 'm': [b1], 'n': [b2]}, {'m': [c1], 'n': [c2]}, x, y, **kwargs)