from .functions import defun, defun_wrapped
@defun_wrapped
def bessely(ctx, n, z, derivative=0, **kwargs):
    if not z:
        if derivative:
            raise ValueError
        if not n:
            return -ctx.inf + (n + z)
        if ctx.im(n):
            return ctx.nan * (n + z)
        r = ctx.re(n)
        q = n + 0.5
        if ctx.isint(q):
            if n > 0:
                return -ctx.inf + (n + z)
            else:
                return 0 * (n + z)
        if r < 0 and int(ctx.floor(q)) % 2:
            return ctx.inf + (n + z)
        else:
            return ctx.ninf + (n + z)
    ctx.prec += 10
    m, d = ctx.nint_distance(n)
    if d < -ctx.prec:
        h = +ctx.eps
        ctx.prec *= 2
        n += h
    elif d < 0:
        ctx.prec -= d
    cos, sin = ctx.cospi_sinpi(n)
    return (ctx.besselj(n, z, derivative, **kwargs) * cos - ctx.besselj(-n, z, derivative, **kwargs)) / sin