from sympy.core.numbers import (oo, pi)
from sympy.core.symbol import Wild
from sympy.core.expr import Expr
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import sin, cos, sinc
from sympy.series.series_class import SeriesBase
from sympy.series.sequences import SeqFormula
from sympy.sets.sets import Interval
from sympy.utilities.iterables import is_sequence
def fourier_series(f, limits=None, finite=True):
    """Computes the Fourier trigonometric series expansion.

    Explanation
    ===========

    Fourier trigonometric series of $f(x)$ over the interval $(a, b)$
    is defined as:

    .. math::
        \\frac{a_0}{2} + \\sum_{n=1}^{\\infty}
        (a_n \\cos(\\frac{2n \\pi x}{L}) + b_n \\sin(\\frac{2n \\pi x}{L}))

    where the coefficients are:

    .. math::
        L = b - a

    .. math::
        a_0 = \\frac{2}{L} \\int_{a}^{b}{f(x) dx}

    .. math::
        a_n = \\frac{2}{L} \\int_{a}^{b}{f(x) \\cos(\\frac{2n \\pi x}{L}) dx}

    .. math::
        b_n = \\frac{2}{L} \\int_{a}^{b}{f(x) \\sin(\\frac{2n \\pi x}{L}) dx}

    The condition whether the function $f(x)$ given should be periodic
    or not is more than necessary, because it is sufficient to consider
    the series to be converging to $f(x)$ only in the given interval,
    not throughout the whole real line.

    This also brings a lot of ease for the computation because
    you do not have to make $f(x)$ artificially periodic by
    wrapping it with piecewise, modulo operations,
    but you can shape the function to look like the desired periodic
    function only in the interval $(a, b)$, and the computed series will
    automatically become the series of the periodic version of $f(x)$.

    This property is illustrated in the examples section below.

    Parameters
    ==========

    limits : (sym, start, end), optional
        *sym* denotes the symbol the series is computed with respect to.

        *start* and *end* denotes the start and the end of the interval
        where the fourier series converges to the given function.

        Default range is specified as $-\\pi$ and $\\pi$.

    Returns
    =======

    FourierSeries
        A symbolic object representing the Fourier trigonometric series.

    Examples
    ========

    Computing the Fourier series of $f(x) = x^2$:

    >>> from sympy import fourier_series, pi
    >>> from sympy.abc import x
    >>> f = x**2
    >>> s = fourier_series(f, (x, -pi, pi))
    >>> s1 = s.truncate(n=3)
    >>> s1
    -4*cos(x) + cos(2*x) + pi**2/3

    Shifting of the Fourier series:

    >>> s.shift(1).truncate()
    -4*cos(x) + cos(2*x) + 1 + pi**2/3
    >>> s.shiftx(1).truncate()
    -4*cos(x + 1) + cos(2*x + 2) + pi**2/3

    Scaling of the Fourier series:

    >>> s.scale(2).truncate()
    -8*cos(x) + 2*cos(2*x) + 2*pi**2/3
    >>> s.scalex(2).truncate()
    -4*cos(2*x) + cos(4*x) + pi**2/3

    Computing the Fourier series of $f(x) = x$:

    This illustrates how truncating to the higher order gives better
    convergence.

    .. plot::
        :context: reset
        :format: doctest
        :include-source: True

        >>> from sympy import fourier_series, pi, plot
        >>> from sympy.abc import x
        >>> f = x
        >>> s = fourier_series(f, (x, -pi, pi))
        >>> s1 = s.truncate(n = 3)
        >>> s2 = s.truncate(n = 5)
        >>> s3 = s.truncate(n = 7)
        >>> p = plot(f, s1, s2, s3, (x, -pi, pi), show=False, legend=True)

        >>> p[0].line_color = (0, 0, 0)
        >>> p[0].label = 'x'
        >>> p[1].line_color = (0.7, 0.7, 0.7)
        >>> p[1].label = 'n=3'
        >>> p[2].line_color = (0.5, 0.5, 0.5)
        >>> p[2].label = 'n=5'
        >>> p[3].line_color = (0.3, 0.3, 0.3)
        >>> p[3].label = 'n=7'

        >>> p.show()

    This illustrates how the series converges to different sawtooth
    waves if the different ranges are specified.

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> s1 = fourier_series(x, (x, -1, 1)).truncate(10)
        >>> s2 = fourier_series(x, (x, -pi, pi)).truncate(10)
        >>> s3 = fourier_series(x, (x, 0, 1)).truncate(10)
        >>> p = plot(x, s1, s2, s3, (x, -5, 5), show=False, legend=True)

        >>> p[0].line_color = (0, 0, 0)
        >>> p[0].label = 'x'
        >>> p[1].line_color = (0.7, 0.7, 0.7)
        >>> p[1].label = '[-1, 1]'
        >>> p[2].line_color = (0.5, 0.5, 0.5)
        >>> p[2].label = '[-pi, pi]'
        >>> p[3].line_color = (0.3, 0.3, 0.3)
        >>> p[3].label = '[0, 1]'

        >>> p.show()

    Notes
    =====

    Computing Fourier series can be slow
    due to the integration required in computing
    an, bn.

    It is faster to compute Fourier series of a function
    by using shifting and scaling on an already
    computed Fourier series rather than computing
    again.

    e.g. If the Fourier series of ``x**2`` is known
    the Fourier series of ``x**2 - 1`` can be found by shifting by ``-1``.

    See Also
    ========

    sympy.series.fourier.FourierSeries

    References
    ==========

    .. [1] https://mathworld.wolfram.com/FourierSeries.html
    """
    f = sympify(f)
    limits = _process_limits(f, limits)
    x = limits[0]
    if x not in f.free_symbols:
        return f
    if finite:
        L = abs(limits[2] - limits[1]) / 2
        is_finite, res_f = finite_check(f, x, L)
        if is_finite:
            return FiniteFourierSeries(f, limits, res_f)
    n = Dummy('n')
    center = (limits[1] + limits[2]) / 2
    if center.is_zero:
        neg_f = f.subs(x, -x)
        if f == neg_f:
            a0, an = fourier_cos_seq(f, limits, n)
            bn = SeqFormula(0, (1, oo))
            return FourierSeries(f, limits, (a0, an, bn))
        elif f == -neg_f:
            a0 = S.Zero
            an = SeqFormula(0, (1, oo))
            bn = fourier_sin_seq(f, limits, n)
            return FourierSeries(f, limits, (a0, an, bn))
    a0, an = fourier_cos_seq(f, limits, n)
    bn = fourier_sin_seq(f, limits, n)
    return FourierSeries(f, limits, (a0, an, bn))