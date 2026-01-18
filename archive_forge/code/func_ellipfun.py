from .functions import defun, defun_wrapped
@defun
def ellipfun(ctx, kind, u=None, m=None, q=None, k=None, tau=None):
    try:
        S = jacobi_spec[kind]
    except KeyError:
        raise ValueError("First argument must be a two-character string containing 's', 'c', 'd' or 'n', e.g.: 'sn'")
    if u is None:

        def f(*args, **kwargs):
            return ctx.ellipfun(kind, *args, **kwargs)
        f.__name__ = kind
        return f
    prec = ctx.prec
    try:
        ctx.prec += 10
        u = ctx.convert(u)
        q = ctx.qfrom(m=m, q=q, k=k, tau=tau)
        if S is None:
            v = ctx.one + 0 * q * u
        elif q == ctx.zero:
            if S[4] == '1':
                v = ctx.one
            else:
                v = getattr(ctx, S[4])(u)
            v += 0 * q * u
        elif q == ctx.one:
            if S[5] == '1':
                v = ctx.one
            else:
                v = getattr(ctx, S[5])(u)
            v += 0 * q * u
        else:
            t = u / ctx.jtheta(3, 0, q) ** 2
            v = ctx.one
            for a in S[0]:
                v *= ctx.jtheta(a, 0, q)
            for b in S[1]:
                v /= ctx.jtheta(b, 0, q)
            for c in S[2]:
                v *= ctx.jtheta(c, t, q)
            for d in S[3]:
                v /= ctx.jtheta(d, t, q)
    finally:
        ctx.prec = prec
    return +v