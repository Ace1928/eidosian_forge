from .functions import defun, defun_wrapped
@defun_wrapped
def coulombf(ctx, l, eta, z, w=1, chop=True, **kwargs):

    def h(l, eta):
        try:
            jw = ctx.j * w
            jwz = ctx.fmul(jw, z, exact=True)
            jwz2 = ctx.fmul(jwz, -2, exact=True)
            C = ctx.coulombc(l, eta)
            T1 = ([C, z, ctx.exp(jwz)], [1, l + 1, 1], [], [], [1 + l + jw * eta], [2 * l + 2], jwz2)
        except ValueError:
            T1 = ([0], [-1], [], [], [], [], 0)
        return (T1,)
    v = ctx.hypercomb(h, [l, eta], **kwargs)
    if chop and (not ctx.im(l)) and (not ctx.im(eta)) and (not ctx.im(z)) and (ctx.re(z) >= 0):
        v = ctx.re(v)
    return v