from ..libmp.backend import xrange
from .calculus import defun
def fdiff_inner(*f_args):

    def inner(t):
        return f(*f_args[:i] + (t,) + f_args[i + 1:])
    return ctx.diff(inner, f_args[i], order, **options)