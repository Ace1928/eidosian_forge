from .functions import defun, defun_wrapped
@defun
def square_exp_arg(ctx, z, mult=1, reciprocal=False):
    prec = ctx.prec * 4 + 20
    if reciprocal:
        z2 = ctx.fmul(z, z, prec=prec)
        z2 = ctx.fdiv(ctx.one, z2, prec=prec)
    else:
        z2 = ctx.fmul(z, z, prec=prec)
    if mult != 1:
        z2 = ctx.fmul(z2, mult, exact=True)
    return z2