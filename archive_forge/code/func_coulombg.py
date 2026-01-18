from .functions import defun, defun_wrapped
@defun_wrapped
def coulombg(ctx, l, eta, z, w=1, chop=True, **kwargs):
    if not ctx._im(l):
        l = ctx._re(l)

    def h(l, eta):
        if ctx.isint(l * 2):
            T1 = ([0], [-1], [], [], [], [], 0)
            return (T1,)
        l2 = -l - 1
        try:
            chi = ctx._coulomb_chi(l, eta)
            jw = ctx.j * w
            s = ctx.sin(chi)
            c = ctx.cos(chi)
            C1 = ctx.coulombc(l, eta)
            C2 = ctx.coulombc(l2, eta)
            u = ctx.exp(jw * z)
            x = -2 * jw * z
            T1 = ([s, C1, z, u, c], [-1, 1, l + 1, 1, 1], [], [], [1 + l + jw * eta], [2 * l + 2], x)
            T2 = ([-s, C2, z, u], [-1, 1, l2 + 1, 1], [], [], [1 + l2 + jw * eta], [2 * l2 + 2], x)
            return (T1, T2)
        except ValueError:
            T1 = ([0], [-1], [], [], [], [], 0)
            return (T1,)
    v = ctx.hypercomb(h, [l, eta], **kwargs)
    if chop and (not ctx._im(l)) and (not ctx._im(eta)) and (not ctx._im(z)) and (ctx._re(z) >= 0):
        v = ctx._re(v)
    return v