from ..libmp.backend import xrange
from .calculus import defun
@defun
def fourier(ctx, f, interval, N):
    """
    Computes the Fourier series of degree `N` of the given function
    on the interval `[a, b]`. More precisely, :func:`~mpmath.fourier` returns
    two lists `(c, s)` of coefficients (the cosine series and sine
    series, respectively), such that

    .. math ::

        f(x) \\sim \\sum_{k=0}^N
            c_k \\cos(k m x) + s_k \\sin(k m x)

    where `m = 2 \\pi / (b-a)`.

    Note that many texts define the first coefficient as `2 c_0` instead
    of `c_0`. The easiest way to evaluate the computed series correctly
    is to pass it to :func:`~mpmath.fourierval`.

    **Examples**

    The function `f(x) = x` has a simple Fourier series on the standard
    interval `[-\\pi, \\pi]`. The cosine coefficients are all zero (because
    the function has odd symmetry), and the sine coefficients are
    rational numbers::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> c, s = fourier(lambda x: x, [-pi, pi], 5)
        >>> nprint(c)
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        >>> nprint(s)
        [0.0, 2.0, -1.0, 0.666667, -0.5, 0.4]

    This computes a Fourier series of a nonsymmetric function on
    a nonstandard interval::

        >>> I = [-1, 1.5]
        >>> f = lambda x: x**2 - 4*x + 1
        >>> cs = fourier(f, I, 4)
        >>> nprint(cs[0])
        [0.583333, 1.12479, -1.27552, 0.904708, -0.441296]
        >>> nprint(cs[1])
        [0.0, -2.6255, 0.580905, 0.219974, -0.540057]

    It is instructive to plot a function along with its truncated
    Fourier series::

        >>> plot([f, lambda x: fourierval(cs, I, x)], I) #doctest: +SKIP

    Fourier series generally converge slowly (and may not converge
    pointwise). For example, if `f(x) = \\cosh(x)`, a 10-term Fourier
    series gives an `L^2` error corresponding to 2-digit accuracy::

        >>> I = [-1, 1]
        >>> cs = fourier(cosh, I, 9)
        >>> g = lambda x: (cosh(x) - fourierval(cs, I, x))**2
        >>> nprint(sqrt(quad(g, I)))
        0.00467963

    :func:`~mpmath.fourier` uses numerical quadrature. For nonsmooth functions,
    the accuracy (and speed) can be improved by including all singular
    points in the interval specification::

        >>> nprint(fourier(abs, [-1, 1], 0), 10)
        ([0.5000441648], [0.0])
        >>> nprint(fourier(abs, [-1, 0, 1], 0), 10)
        ([0.5], [0.0])

    """
    interval = ctx._as_points(interval)
    a = interval[0]
    b = interval[-1]
    L = b - a
    cos_series = []
    sin_series = []
    cutoff = ctx.eps * 10
    for n in xrange(N + 1):
        m = 2 * n * ctx.pi / L
        an = 2 * ctx.quadgl(lambda t: f(t) * ctx.cos(m * t), interval) / L
        bn = 2 * ctx.quadgl(lambda t: f(t) * ctx.sin(m * t), interval) / L
        if n == 0:
            an /= 2
        if abs(an) < cutoff:
            an = ctx.zero
        if abs(bn) < cutoff:
            bn = ctx.zero
        cos_series.append(an)
        sin_series.append(bn)
    return (cos_series, sin_series)