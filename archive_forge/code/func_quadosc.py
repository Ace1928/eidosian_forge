import math
from ..libmp.backend import xrange
def quadosc(ctx, f, interval, omega=None, period=None, zeros=None):
    """
        Calculates

        .. math ::

            I = \\int_a^b f(x) dx

        where at least one of `a` and `b` is infinite and where
        `f(x) = g(x) \\cos(\\omega x  + \\phi)` for some slowly
        decreasing function `g(x)`. With proper input, :func:`~mpmath.quadosc`
        can also handle oscillatory integrals where the oscillation
        rate is different from a pure sine or cosine wave.

        In the standard case when `|a| < \\infty, b = \\infty`,
        :func:`~mpmath.quadosc` works by evaluating the infinite series

        .. math ::

            I = \\int_a^{x_1} f(x) dx +
            \\sum_{k=1}^{\\infty} \\int_{x_k}^{x_{k+1}} f(x) dx

        where `x_k` are consecutive zeros (alternatively
        some other periodic reference point) of `f(x)`.
        Accordingly, :func:`~mpmath.quadosc` requires information about the
        zeros of `f(x)`. For a periodic function, you can specify
        the zeros by either providing the angular frequency `\\omega`
        (*omega*) or the *period* `2 \\pi/\\omega`. In general, you can
        specify the `n`-th zero by providing the *zeros* arguments.
        Below is an example of each::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = True
            >>> f = lambda x: sin(3*x)/(x**2+1)
            >>> quadosc(f, [0,inf], omega=3)
            0.37833007080198
            >>> quadosc(f, [0,inf], period=2*pi/3)
            0.37833007080198
            >>> quadosc(f, [0,inf], zeros=lambda n: pi*n/3)
            0.37833007080198
            >>> (ei(3)*exp(-3)-exp(3)*ei(-3))/2  # Computed by Mathematica
            0.37833007080198

        Note that *zeros* was specified to multiply `n` by the
        *half-period*, not the full period. In theory, it does not matter
        whether each partial integral is done over a half period or a full
        period. However, if done over half-periods, the infinite series
        passed to :func:`~mpmath.nsum` becomes an *alternating series* and this
        typically makes the extrapolation much more efficient.

        Here is an example of an integration over the entire real line,
        and a half-infinite integration starting at `-\\infty`::

            >>> quadosc(lambda x: cos(x)/(1+x**2), [-inf, inf], omega=1)
            1.15572734979092
            >>> pi/e
            1.15572734979092
            >>> quadosc(lambda x: cos(x)/x**2, [-inf, -1], period=2*pi)
            -0.0844109505595739
            >>> cos(1)+si(1)-pi/2
            -0.0844109505595738

        Of course, the integrand may contain a complex exponential just as
        well as a real sine or cosine::

            >>> quadosc(lambda x: exp(3*j*x)/(1+x**2), [-inf,inf], omega=3)
            (0.156410688228254 + 0.0j)
            >>> pi/e**3
            0.156410688228254
            >>> quadosc(lambda x: exp(3*j*x)/(2+x+x**2), [-inf,inf], omega=3)
            (0.00317486988463794 - 0.0447701735209082j)
            >>> 2*pi/sqrt(7)/exp(3*(j+sqrt(7))/2)
            (0.00317486988463794 - 0.0447701735209082j)

        **Non-periodic functions**

        If `f(x) = g(x) h(x)` for some function `h(x)` that is not
        strictly periodic, *omega* or *period* might not work, and it might
        be necessary to use *zeros*.

        A notable exception can be made for Bessel functions which, though not
        periodic, are "asymptotically periodic" in a sufficiently strong sense
        that the sum extrapolation will work out::

            >>> quadosc(j0, [0, inf], period=2*pi)
            1.0
            >>> quadosc(j1, [0, inf], period=2*pi)
            1.0

        More properly, one should provide the exact Bessel function zeros::

            >>> j0zero = lambda n: findroot(j0, pi*(n-0.25))
            >>> quadosc(j0, [0, inf], zeros=j0zero)
            1.0

        For an example where *zeros* becomes necessary, consider the
        complete Fresnel integrals

        .. math ::

            \\int_0^{\\infty} \\cos x^2\\,dx = \\int_0^{\\infty} \\sin x^2\\,dx
            = \\sqrt{\\frac{\\pi}{8}}.

        Although the integrands do not decrease in magnitude as
        `x \\to \\infty`, the integrals are convergent since the oscillation
        rate increases (causing consecutive periods to asymptotically
        cancel out). These integrals are virtually impossible to calculate
        to any kind of accuracy using standard quadrature rules. However,
        if one provides the correct asymptotic distribution of zeros
        (`x_n \\sim \\sqrt{n}`), :func:`~mpmath.quadosc` works::

            >>> mp.dps = 30
            >>> f = lambda x: cos(x**2)
            >>> quadosc(f, [0,inf], zeros=lambda n:sqrt(pi*n))
            0.626657068657750125603941321203
            >>> f = lambda x: sin(x**2)
            >>> quadosc(f, [0,inf], zeros=lambda n:sqrt(pi*n))
            0.626657068657750125603941321203
            >>> sqrt(pi/8)
            0.626657068657750125603941321203

        (Interestingly, these integrals can still be evaluated if one
        places some other constant than `\\pi` in the square root sign.)

        In general, if `f(x) \\sim g(x) \\cos(h(x))`, the zeros follow
        the inverse-function distribution `h^{-1}(x)`::

            >>> mp.dps = 15
            >>> f = lambda x: sin(exp(x))
            >>> quadosc(f, [1,inf], zeros=lambda n: log(n))
            -0.25024394235267
            >>> pi/2-si(e)
            -0.250243942352671

        **Non-alternating functions**

        If the integrand oscillates around a positive value, without
        alternating signs, the extrapolation might fail. A simple trick
        that sometimes works is to multiply or divide the frequency by 2::

            >>> f = lambda x: 1/x**2+sin(x)/x**4
            >>> quadosc(f, [1,inf], omega=1)  # Bad
            1.28642190869861
            >>> quadosc(f, [1,inf], omega=0.5)  # Perfect
            1.28652953559617
            >>> 1+(cos(1)+ci(1)+sin(1))/6
            1.28652953559617

        **Fast decay**

        :func:`~mpmath.quadosc` is primarily useful for slowly decaying
        integrands. If the integrand decreases exponentially or faster,
        :func:`~mpmath.quad` will likely handle it without trouble (and generally be
        much faster than :func:`~mpmath.quadosc`)::

            >>> quadosc(lambda x: cos(x)/exp(x), [0, inf], omega=1)
            0.5
            >>> quad(lambda x: cos(x)/exp(x), [0, inf])
            0.5

        """
    a, b = ctx._as_points(interval)
    a = ctx.convert(a)
    b = ctx.convert(b)
    if [omega, period, zeros].count(None) != 2:
        raise ValueError('must specify exactly one of omega, period, zeros')
    if a == ctx.ninf and b == ctx.inf:
        s1 = ctx.quadosc(f, [a, 0], omega=omega, zeros=zeros, period=period)
        s2 = ctx.quadosc(f, [0, b], omega=omega, zeros=zeros, period=period)
        return s1 + s2
    if a == ctx.ninf:
        if zeros:
            return ctx.quadosc(lambda x: f(-x), [-b, -a], lambda n: zeros(-n))
        else:
            return ctx.quadosc(lambda x: f(-x), [-b, -a], omega=omega, period=period)
    if b != ctx.inf:
        raise ValueError('quadosc requires an infinite integration interval')
    if not zeros:
        if omega:
            period = 2 * ctx.pi / omega
        zeros = lambda n: n * period / 2
    n = 1
    s = ctx.quadgl(f, [a, zeros(n)])

    def term(k):
        return ctx.quadgl(f, [zeros(k), zeros(k + 1)])
    s += ctx.nsum(term, [n, ctx.inf])
    return s