from .functions import defun, defun_wrapped
def compute_triple_tvb(ctx, n):
    t = ctx.grampoint(n)
    v = ctx._fp.siegelz(t)
    if ctx.mag(abs(v)) < ctx.mag(t) - 45:
        v = ctx.siegelz(t)
    b = v * (-1) ** n
    return (t, v, b)