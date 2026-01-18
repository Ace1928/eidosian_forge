from .functions import defun, defun_wrapped
@defun
def struvel(ctx, n, z, **kwargs):
    n = ctx.convert(n)
    z = ctx.convert(z)

    def h(n):
        return [([z / 2, 0.5 * ctx.sqrt(ctx.pi)], [n + 1, -1], [], [n + 1.5], [1], [1.5, n + 1.5], (z / 2) ** 2)]
    return ctx.hypercomb(h, [n], **kwargs)