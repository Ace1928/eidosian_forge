from .functions import defun, defun_wrapped
@defun
def e1(ctx, z):
    try:
        return ctx._e1(z)
    except NotImplementedError:
        return ctx.expint(1, z)