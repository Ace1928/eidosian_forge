from .functions import defun, defun_wrapped
@defun
def ber(ctx, n, z, **kwargs):
    n = ctx.convert(n)
    z = ctx.convert(z)

    def h(n):
        r = -(z / 4) ** 4
        cos, sin = ctx.cospi_sinpi(-0.75 * n)
        T1 = ([cos, z / 2], [1, n], [], [n + 1], [], [0.5, 0.5 * (n + 1), 0.5 * n + 1], r)
        T2 = ([sin, z / 2], [1, n + 2], [], [n + 2], [], [1.5, 0.5 * (n + 3), 0.5 * n + 1], r)
        return (T1, T2)
    return ctx.hypercomb(h, [n], **kwargs)