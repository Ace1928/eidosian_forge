from ..libmp.backend import xrange
from .calculus import defun
def cohen_alt(ctx):
    L = cohen_alt_class()
    L.ctx = ctx
    return L