from ..libmp.backend import xrange
from .calculus import defun
@defun
def diffs_prod(ctx, factors):
    """
    Given a list of `N` iterables or generators yielding
    `f_k(x), f'_k(x), f''_k(x), \\ldots` for `k = 1, \\ldots, N`,
    generate `g(x), g'(x), g''(x), \\ldots` where
    `g(x) = f_1(x) f_2(x) \\cdots f_N(x)`.

    At high precision and for large orders, this is typically more efficient
    than numerical differentiation if the derivatives of each `f_k(x)`
    admit direct computation.

    Note: This function does not increase the working precision internally,
    so guard digits may have to be added externally for full accuracy.

    **Examples**

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> f = lambda x: exp(x)*cos(x)*sin(x)
        >>> u = diffs(f, 1)
        >>> v = mp.diffs_prod([diffs(exp,1), diffs(cos,1), diffs(sin,1)])
        >>> next(u); next(v)
        1.23586333600241
        1.23586333600241
        >>> next(u); next(v)
        0.104658952245596
        0.104658952245596
        >>> next(u); next(v)
        -5.96999877552086
        -5.96999877552086
        >>> next(u); next(v)
        -12.4632923122697
        -12.4632923122697

    """
    N = len(factors)
    if N == 1:
        for c in factors[0]:
            yield c
    else:
        u = iterable_to_function(ctx.diffs_prod(factors[:N // 2]))
        v = iterable_to_function(ctx.diffs_prod(factors[N // 2:]))
        n = 0
        while 1:
            s = u(n) * v(0)
            a = 1
            for k in xrange(1, n + 1):
                a = a * (n - k + 1) // k
                s += a * u(n - k) * v(k)
            yield s
            n += 1