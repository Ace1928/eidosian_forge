from .functions import defun, defun_wrapped
@defun
def ker(ctx, n, z, **kwargs):
    n = ctx.convert(n)
    z = ctx.convert(z)

    def h(n):
        r = -(z / 4) ** 4
        cos1, sin1 = ctx.cospi_sinpi(0.25 * n)
        cos2, sin2 = ctx.cospi_sinpi(0.75 * n)
        T1 = ([2, z, 4 * cos1], [-n - 3, n, 1], [-n], [], [], [0.5, 0.5 * (1 + n), 0.5 * (n + 2)], r)
        T2 = ([2, z, -sin1], [-n - 3, 2 + n, 1], [-n - 1], [], [], [1.5, 0.5 * (3 + n), 0.5 * (n + 2)], r)
        T3 = ([2, z, 4 * cos2], [n - 3, -n, 1], [n], [], [], [0.5, 0.5 * (1 - n), 1 - 0.5 * n], r)
        T4 = ([2, z, -sin2], [n - 3, 2 - n, 1], [n - 1], [], [], [1.5, 0.5 * (3 - n), 1 - 0.5 * n], r)
        return (T1, T2, T3, T4)
    return ctx.hypercomb(h, [n], **kwargs)