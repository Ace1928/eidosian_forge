from ..libmp.backend import xrange
from .calculus import defun
@defun
def diffs_exp(ctx, fdiffs):
    """
    Given an iterable or generator yielding `f(x), f'(x), f''(x), \\ldots`
    generate `g(x), g'(x), g''(x), \\ldots` where `g(x) = \\exp(f(x))`.

    At high precision and for large orders, this is typically more efficient
    than numerical differentiation if the derivatives of `f(x)`
    admit direct computation.

    Note: This function does not increase the working precision internally,
    so guard digits may have to be added externally for full accuracy.

    **Examples**

    The derivatives of the gamma function can be computed using
    logarithmic differentiation::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>>
        >>> def diffs_loggamma(x):
        ...     yield loggamma(x)
        ...     i = 0
        ...     while 1:
        ...         yield psi(i,x)
        ...         i += 1
        ...
        >>> u = diffs_exp(diffs_loggamma(3))
        >>> v = diffs(gamma, 3)
        >>> next(u); next(v)
        2.0
        2.0
        >>> next(u); next(v)
        1.84556867019693
        1.84556867019693
        >>> next(u); next(v)
        2.49292999190269
        2.49292999190269
        >>> next(u); next(v)
        3.44996501352367
        3.44996501352367

    """
    fn = iterable_to_function(fdiffs)
    f0 = ctx.exp(fn(0))
    yield f0
    i = 1
    while 1:
        s = ctx.mpf(0)
        for powers, c in iteritems(dpoly(i)):
            s += c * ctx.fprod((fn(k + 1) ** p for k, p in enumerate(powers) if p))
        yield (s * f0)
        i += 1