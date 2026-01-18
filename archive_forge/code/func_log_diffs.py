from ..libmp.backend import xrange
from .functions import defun, defun_wrapped
def log_diffs(k0):
    b2 = b_s + [1]
    yield (sum((ctx.loggamma(a + k0) for a in a_s)) - sum((ctx.loggamma(b + k0) for b in b2)) + k0 * ctx.log(z))
    i = 0
    while 1:
        v = sum((ctx.psi(i, a + k0) for a in a_s)) - sum((ctx.psi(i, b + k0) for b in b2))
        if i == 0:
            v += ctx.log(z)
        yield v
        i += 1