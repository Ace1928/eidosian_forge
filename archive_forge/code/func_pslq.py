from .libmp.backend import xrange
from .libmp import int_types, sqrt_fixed
def pslq(ctx, x, tol=None, maxcoeff=1000, maxsteps=100, verbose=False):
    """
    Given a vector of real numbers `x = [x_0, x_1, ..., x_n]`, ``pslq(x)``
    uses the PSLQ algorithm to find a list of integers
    `[c_0, c_1, ..., c_n]` such that

    .. math ::

        |c_1 x_1 + c_2 x_2 + ... + c_n x_n| < \\mathrm{tol}

    and such that `\\max |c_k| < \\mathrm{maxcoeff}`. If no such vector
    exists, :func:`~mpmath.pslq` returns ``None``. The tolerance defaults to
    3/4 of the working precision.

    **Examples**

    Find rational approximations for `\\pi`::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> pslq([-1, pi], tol=0.01)
        [22, 7]
        >>> pslq([-1, pi], tol=0.001)
        [355, 113]
        >>> mpf(22)/7; mpf(355)/113; +pi
        3.14285714285714
        3.14159292035398
        3.14159265358979

    Pi is not a rational number with denominator less than 1000::

        >>> pslq([-1, pi])
        >>>

    To within the standard precision, it can however be approximated
    by at least one rational number with denominator less than `10^{12}`::

        >>> p, q = pslq([-1, pi], maxcoeff=10**12)
        >>> print(p); print(q)
        238410049439
        75888275702
        >>> mpf(p)/q
        3.14159265358979

    The PSLQ algorithm can be applied to long vectors. For example,
    we can investigate the rational (in)dependence of integer square
    roots::

        >>> mp.dps = 30
        >>> pslq([sqrt(n) for n in range(2, 5+1)])
        >>>
        >>> pslq([sqrt(n) for n in range(2, 6+1)])
        >>>
        >>> pslq([sqrt(n) for n in range(2, 8+1)])
        [2, 0, 0, 0, 0, 0, -1]

    **Machin formulas**

    A famous formula for `\\pi` is Machin's,

    .. math ::

        \\frac{\\pi}{4} = 4 \\operatorname{acot} 5 - \\operatorname{acot} 239

    There are actually infinitely many formulas of this type. Two
    others are

    .. math ::

        \\frac{\\pi}{4} = \\operatorname{acot} 1

        \\frac{\\pi}{4} = 12 \\operatorname{acot} 49 + 32 \\operatorname{acot} 57
            + 5 \\operatorname{acot} 239 + 12 \\operatorname{acot} 110443

    We can easily verify the formulas using the PSLQ algorithm::

        >>> mp.dps = 30
        >>> pslq([pi/4, acot(1)])
        [1, -1]
        >>> pslq([pi/4, acot(5), acot(239)])
        [1, -4, 1]
        >>> pslq([pi/4, acot(49), acot(57), acot(239), acot(110443)])
        [1, -12, -32, 5, -12]

    We could try to generate a custom Machin-like formula by running
    the PSLQ algorithm with a few inverse cotangent values, for example
    acot(2), acot(3) ... acot(10). Unfortunately, there is a linear
    dependence among these values, resulting in only that dependence
    being detected, with a zero coefficient for `\\pi`::

        >>> pslq([pi] + [acot(n) for n in range(2,11)])
        [0, 1, -1, 0, 0, 0, -1, 0, 0, 0]

    We get better luck by removing linearly dependent terms::

        >>> pslq([pi] + [acot(n) for n in range(2,11) if n not in (3, 5)])
        [1, -8, 0, 0, 4, 0, 0, 0]

    In other words, we found the following formula::

        >>> 8*acot(2) - 4*acot(7)
        3.14159265358979323846264338328
        >>> +pi
        3.14159265358979323846264338328

    **Algorithm**

    This is a fairly direct translation to Python of the pseudocode given by
    David Bailey, "The PSLQ Integer Relation Algorithm":
    http://www.cecm.sfu.ca/organics/papers/bailey/paper/html/node3.html

    The present implementation uses fixed-point instead of floating-point
    arithmetic, since this is significantly (about 7x) faster.
    """
    n = len(x)
    if n < 2:
        raise ValueError('n cannot be less than 2')
    prec = ctx.prec
    if prec < 53:
        raise ValueError('prec cannot be less than 53')
    if verbose and prec // max(2, n) < 5:
        print('Warning: precision for PSLQ may be too low')
    target = int(prec * 0.75)
    if tol is None:
        tol = ctx.mpf(2) ** (-target)
    else:
        tol = ctx.convert(tol)
    extra = 60
    prec += extra
    if verbose:
        print('PSLQ using prec %i and tol %s' % (prec, ctx.nstr(tol)))
    tol = ctx.to_fixed(tol, prec)
    assert tol
    x = [None] + [ctx.to_fixed(ctx.mpf(xk), prec) for xk in x]
    minx = min((abs(xx) for xx in x[1:]))
    if not minx:
        raise ValueError('PSLQ requires a vector of nonzero numbers')
    if minx < tol // 100:
        if verbose:
            print('STOPPING: (one number is too small)')
        return None
    g = sqrt_fixed((4 << prec) // 3, prec)
    A = {}
    B = {}
    H = {}
    for i in xrange(1, n + 1):
        for j in xrange(1, n + 1):
            A[i, j] = B[i, j] = (i == j) << prec
            H[i, j] = 0
    s = [None] + [0] * n
    for k in xrange(1, n + 1):
        t = 0
        for j in xrange(k, n + 1):
            t += x[j] ** 2 >> prec
        s[k] = sqrt_fixed(t, prec)
    t = s[1]
    y = x[:]
    for k in xrange(1, n + 1):
        y[k] = (x[k] << prec) // t
        s[k] = (s[k] << prec) // t
    for i in xrange(1, n + 1):
        for j in xrange(i + 1, n):
            H[i, j] = 0
        if i <= n - 1:
            if s[i]:
                H[i, i] = (s[i + 1] << prec) // s[i]
            else:
                H[i, i] = 0
        for j in range(1, i):
            sjj1 = s[j] * s[j + 1]
            if sjj1:
                H[i, j] = (-y[i] * y[j] << prec) // sjj1
            else:
                H[i, j] = 0
    for i in xrange(2, n + 1):
        for j in xrange(i - 1, 0, -1):
            if H[j, j]:
                t = round_fixed((H[i, j] << prec) // H[j, j], prec)
            else:
                continue
            y[j] = y[j] + (t * y[i] >> prec)
            for k in xrange(1, j + 1):
                H[i, k] = H[i, k] - (t * H[j, k] >> prec)
            for k in xrange(1, n + 1):
                A[i, k] = A[i, k] - (t * A[j, k] >> prec)
                B[k, j] = B[k, j] + (t * B[k, i] >> prec)
    for REP in range(maxsteps):
        m = -1
        szmax = -1
        for i in range(1, n):
            h = H[i, i]
            sz = g ** i * abs(h) >> prec * (i - 1)
            if sz > szmax:
                m = i
                szmax = sz
        y[m], y[m + 1] = (y[m + 1], y[m])
        for i in xrange(1, n + 1):
            H[m, i], H[m + 1, i] = (H[m + 1, i], H[m, i])
        for i in xrange(1, n + 1):
            A[m, i], A[m + 1, i] = (A[m + 1, i], A[m, i])
        for i in xrange(1, n + 1):
            B[i, m], B[i, m + 1] = (B[i, m + 1], B[i, m])
        if m <= n - 2:
            t0 = sqrt_fixed(H[m, m] ** 2 + H[m, m + 1] ** 2 >> prec, prec)
            if not t0:
                break
            t1 = (H[m, m] << prec) // t0
            t2 = (H[m, m + 1] << prec) // t0
            for i in xrange(m, n + 1):
                t3 = H[i, m]
                t4 = H[i, m + 1]
                H[i, m] = t1 * t3 + t2 * t4 >> prec
                H[i, m + 1] = -t2 * t3 + t1 * t4 >> prec
        for i in xrange(m + 1, n + 1):
            for j in xrange(min(i - 1, m + 1), 0, -1):
                try:
                    t = round_fixed((H[i, j] << prec) // H[j, j], prec)
                except ZeroDivisionError:
                    break
                y[j] = y[j] + (t * y[i] >> prec)
                for k in xrange(1, j + 1):
                    H[i, k] = H[i, k] - (t * H[j, k] >> prec)
                for k in xrange(1, n + 1):
                    A[i, k] = A[i, k] - (t * A[j, k] >> prec)
                    B[k, j] = B[k, j] + (t * B[k, i] >> prec)
        best_err = maxcoeff << prec
        for i in xrange(1, n + 1):
            err = abs(y[i])
            if err < tol:
                vec = [int(round_fixed(B[j, i], prec) >> prec) for j in range(1, n + 1)]
                if max((abs(v) for v in vec)) < maxcoeff:
                    if verbose:
                        print('FOUND relation at iter %i/%i, error: %s' % (REP, maxsteps, ctx.nstr(err / ctx.mpf(2) ** prec, 1)))
                    return vec
            best_err = min(err, best_err)
        recnorm = max((abs(h) for h in H.values()))
        if recnorm:
            norm = (1 << 2 * prec) // recnorm >> prec
            norm //= 100
        else:
            norm = ctx.inf
        if verbose:
            print('%i/%i:  Error: %8s   Norm: %s' % (REP, maxsteps, ctx.nstr(best_err / ctx.mpf(2) ** prec, 1), norm))
        if norm >= maxcoeff:
            break
    if verbose:
        print('CANCELLING after step %i/%i.' % (REP, maxsteps))
        print('Could not find an integer relation. Norm bound: %s' % norm)
    return None