from .functions import defun, defun_wrapped
def bessel_zero(ctx, kind, prime, v, m, isoltol=0.01, _interval_cache={}):
    prec = ctx.prec
    workprec = max(prec, ctx.mag(v), ctx.mag(m)) + 10
    try:
        ctx.prec = workprec
        v = ctx.mpf(v)
        m = int(m)
        prime = int(prime)
        if v < 0:
            raise ValueError('v cannot be negative')
        if m < 1:
            raise ValueError('m cannot be less than 1')
        if not prime in (0, 1):
            raise ValueError('prime should lie between 0 and 1')
        if kind == 1:
            if prime:
                f = lambda x: ctx.besselj(v, x, derivative=1)
            else:
                f = lambda x: ctx.besselj(v, x)
        if kind == 2:
            if prime:
                f = lambda x: ctx.bessely(v, x, derivative=1)
            else:
                f = lambda x: ctx.bessely(v, x)
        if kind == 1 and prime and (m == 1):
            if v == 0:
                return ctx.zero
            if v <= 1:
                r = 2 * ctx.sqrt(v * (1 + v) / (v + 2))
                return find_in_interval(ctx, f, (r / 10, 2 * r))
        if (kind, prime, v, m) in _interval_cache:
            return find_in_interval(ctx, f, _interval_cache[kind, prime, v, m])
        r, err = mcmahon(ctx, kind, prime, v, m)
        if err < isoltol:
            return find_in_interval(ctx, f, (r - isoltol, r + isoltol))
        if kind == 1 and (not prime):
            low = 2.4
        if kind == 1 and prime:
            low = 1.8
        if kind == 2 and (not prime):
            low = 0.8
        if kind == 2 and prime:
            low = 2.0
        n = m + 1
        while 1:
            r1, err = mcmahon(ctx, kind, prime, v, n)
            if err < isoltol:
                r2, err2 = mcmahon(ctx, kind, prime, v, n + 1)
                intervals = generalized_bisection(ctx, f, low, 0.5 * (r1 + r2), n)
                for k, ab in enumerate(intervals):
                    _interval_cache[kind, prime, v, k + 1] = ab
                return find_in_interval(ctx, f, intervals[m - 1])
            else:
                n = n * 2
    finally:
        ctx.prec = prec