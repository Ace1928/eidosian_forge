from ..libmp.backend import xrange
from .functions import defun, defun_wrapped
@defun
def gammaprod(ctx, a, b, _infsign=False):
    a = [ctx.convert(x) for x in a]
    b = [ctx.convert(x) for x in b]
    poles_num = []
    poles_den = []
    regular_num = []
    regular_den = []
    for x in a:
        [regular_num, poles_num][ctx.isnpint(x)].append(x)
    for x in b:
        [regular_den, poles_den][ctx.isnpint(x)].append(x)
    if len(poles_num) < len(poles_den):
        return ctx.zero
    if len(poles_num) > len(poles_den):
        if _infsign:
            a = [x and x * (1 + ctx.eps) or x + ctx.eps for x in poles_num]
            b = [x and x * (1 + ctx.eps) or x + ctx.eps for x in poles_den]
            return ctx.sign(ctx.gammaprod(a + regular_num, b + regular_den)) * ctx.inf
        else:
            return ctx.inf
    p = ctx.one
    orig = ctx.prec
    try:
        ctx.prec = orig + 15
        while poles_num:
            i = poles_num.pop()
            j = poles_den.pop()
            p *= (-1) ** (i + j) * ctx.gamma(1 - j) / ctx.gamma(1 - i)
        for x in regular_num:
            p *= ctx.gamma(x)
        for x in regular_den:
            p /= ctx.gamma(x)
    finally:
        ctx.prec = orig
    return +p