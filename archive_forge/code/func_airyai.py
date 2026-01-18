from .functions import defun, defun_wrapped
@defun
def airyai(ctx, z, derivative=0, **kwargs):
    z = ctx.convert(z)
    if derivative:
        n, ntype = ctx._convert_param(derivative)
    else:
        n = 0
    if not ctx.isnormal(z) and z:
        if n and ntype == 'Z':
            if n == -1:
                if z == ctx.inf:
                    return ctx.mpf(1) / 3 + 1 / z
                if z == ctx.ninf:
                    return ctx.mpf(-2) / 3 + 1 / z
            if n < -1:
                if z == ctx.inf:
                    return z
                if z == ctx.ninf:
                    return (-1) ** n * -z
        if not n and z == ctx.inf or z == ctx.ninf:
            return 1 / z
        raise ValueError('essential singularity of Ai(z)')
    if z:
        extraprec = max(0, int(1.5 * ctx.mag(z)))
    else:
        extraprec = 0
    if n:
        if n == 1:

            def h():
                if ctx._re(z) > 4:
                    ctx.prec += extraprec
                    w = z ** 1.5
                    r = -0.75 / w
                    u = -2 * w / 3
                    ctx.prec -= extraprec
                    C = -ctx.exp(u) / (2 * ctx.sqrt(ctx.pi)) * ctx.nthroot(z, 4)
                    return (([C], [1], [], [], [(-1, 6), (7, 6)], [], r),)
                else:
                    ctx.prec += extraprec
                    w = z ** 3 / 9
                    ctx.prec -= extraprec
                    C1 = _airyai_C1(ctx) * 0.5
                    C2 = _airyai_C2(ctx)
                    T1 = ([C1, z], [1, 2], [], [], [], [ctx.mpq_5_3], w)
                    T2 = ([C2], [1], [], [], [], [ctx.mpq_1_3], w)
                    return (T1, T2)
            return ctx.hypercomb(h, [], **kwargs)
        else:
            if z == 0:
                return _airyderiv_0(ctx, z, n, ntype, 0)

            def h(n):
                ctx.prec += extraprec
                w = z ** 3 / 9
                ctx.prec -= extraprec
                q13, q23, q43 = (ctx.mpq_1_3, ctx.mpq_2_3, ctx.mpq_4_3)
                a1 = q13
                a2 = 1
                b1 = (1 - n) * q13
                b2 = (2 - n) * q13
                b3 = 1 - n * q13
                T1 = ([3, z], [n - q23, -n], [a1], [b1, b2, b3], [a1, a2], [b1, b2, b3], w)
                a1 = q23
                b1 = (2 - n) * q13
                b2 = 1 - n * q13
                b3 = (4 - n) * q13
                T2 = ([3, z, -z], [n - q43, -n, 1], [a1], [b1, b2, b3], [a1, a2], [b1, b2, b3], w)
                return (T1, T2)
            v = ctx.hypercomb(h, [n], **kwargs)
            if ctx._is_real_type(z) and ctx.isint(n):
                v = ctx._re(v)
            return v
    else:

        def h():
            if ctx._re(z) > 4:
                ctx.prec += extraprec
                w = z ** 1.5
                r = -0.75 / w
                u = -2 * w / 3
                ctx.prec -= extraprec
                C = ctx.exp(u) / (2 * ctx.sqrt(ctx.pi) * ctx.nthroot(z, 4))
                return (([C], [1], [], [], [(1, 6), (5, 6)], [], r),)
            else:
                ctx.prec += extraprec
                w = z ** 3 / 9
                ctx.prec -= extraprec
                C1 = _airyai_C1(ctx)
                C2 = _airyai_C2(ctx)
                T1 = ([C1], [1], [], [], [], [ctx.mpq_2_3], w)
                T2 = ([z * C2], [1], [], [], [], [ctx.mpq_4_3], w)
                return (T1, T2)
        return ctx.hypercomb(h, [], **kwargs)