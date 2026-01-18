from ..libmp.backend import xrange
from .functions import defun, defun_wrapped
@defun
def appellf4(ctx, a, b, c1, c2, x, y, **kwargs):
    return ctx.hyper2d({'m+n': [a, b]}, {'m': [c1], 'n': [c2]}, x, y, **kwargs)