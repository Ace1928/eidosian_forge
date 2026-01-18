import math
from .functions import defun
@defun
def rs_zeta(ctx, s, derivative=0, **kwargs):
    if derivative > 4:
        raise NotImplementedError
    s = ctx.convert(s)
    re = ctx._re(s)
    im = ctx._im(s)
    if im < 0:
        z = ctx.conj(ctx.rs_zeta(ctx.conj(s), derivative))
        return z
    critical_line = re == 0.5
    if critical_line:
        return zeta_half(ctx, s, derivative)
    else:
        return zeta_offline(ctx, s, derivative)