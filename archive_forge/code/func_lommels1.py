from .functions import defun, defun_wrapped
@defun
def lommels1(ctx, u, v, z, **kwargs):
    u = ctx._convert_param(u)[0]
    v = ctx._convert_param(v)[0]
    z = ctx.convert(z)

    def h(u, v):
        b = ctx.mpq_1_2
        w = ctx.square_exp_arg(z, mult=-0.25)
        return (([u - v + 1, u + v + 1, z], [-1, -1, u + 1], [], [], [1], [b * (u - v + 3), b * (u + v + 3)], w),)
    return ctx.hypercomb(h, [u, v], **kwargs)