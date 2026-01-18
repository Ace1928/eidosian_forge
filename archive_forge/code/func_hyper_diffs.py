from ..libmp.backend import xrange
from .functions import defun, defun_wrapped
def hyper_diffs(k0):
    C = ctx.gammaprod([b for b in b_s], [a for a in a_s])
    for d in ctx.diffs_exp(log_diffs(k0)):
        v = C * d
        yield v