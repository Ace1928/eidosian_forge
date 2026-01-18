from ..libmp.backend import xrange
from .functions import defun, defun_wrapped
@defun_wrapped
def appellf1(ctx, a, b1, b2, c, x, y, **kwargs):
    if abs(x) > abs(y):
        x, y = (y, x)
        b1, b2 = (b2, b1)

    def ok(x):
        return abs(x) < 0.99
    if ctx.isnpint(a):
        pass
    elif ctx.isnpint(b1):
        pass
    elif ctx.isnpint(b2):
        x, y, b1, b2 = (y, x, b2, b1)
    elif not ok(x):
        u1 = (x - y) / (x - 1)
        if not ok(u1):
            raise ValueError('Analytic continuation not implemented')
        return (1 - x) ** (-b1) * (1 - y) ** (c - a - b2) * ctx.appellf1(c - a, b1, c - b1 - b2, c, u1, y, **kwargs)
    return ctx.hyper2d({'m+n': [a], 'm': [b1], 'n': [b2]}, {'m+n': [c]}, x, y, **kwargs)