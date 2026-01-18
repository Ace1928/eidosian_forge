import math
from .functions import defun
@defun
def rs_z(ctx, w, derivative=0):
    w = ctx.convert(w)
    re = ctx._re(w)
    im = ctx._im(w)
    if re < 0:
        return rs_z(ctx, -w, derivative)
    critical_line = im == 0
    if critical_line:
        return z_half(ctx, w, derivative)
    else:
        return z_offline(ctx, w, derivative)