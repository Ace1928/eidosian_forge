from ..libmp.backend import xrange
from .calculus import defun
@defun
def nprod(ctx, f, interval, nsum=False, **kwargs):
    """
    Computes the product

    .. math ::

        P = \\prod_{k=a}^b f(k)

    where `(a, b)` = *interval*, and where `a = -\\infty` and/or
    `b = \\infty` are allowed.

    By default, :func:`~mpmath.nprod` uses the same extrapolation methods as
    :func:`~mpmath.nsum`, except applied to the partial products rather than
    partial sums, and the same keyword options as for :func:`~mpmath.nsum` are
    supported. If ``nsum=True``, the product is instead computed via
    :func:`~mpmath.nsum` as

    .. math ::

        P = \\exp\\left( \\sum_{k=a}^b \\log(f(k)) \\right).

    This is slower, but can sometimes yield better results. It is
    also required (and used automatically) when Euler-Maclaurin
    summation is requested.

    **Examples**

    A simple finite product::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> nprod(lambda k: k, [1, 4])
        24.0

    A large number of infinite products have known exact values,
    and can therefore be used as a reference. Most of the following
    examples are taken from MathWorld [1].

    A few infinite products with simple values are::

        >>> 2*nprod(lambda k: (4*k**2)/(4*k**2-1), [1, inf])
        3.141592653589793238462643
        >>> nprod(lambda k: (1+1/k)**2/(1+2/k), [1, inf])
        2.0
        >>> nprod(lambda k: (k**3-1)/(k**3+1), [2, inf])
        0.6666666666666666666666667
        >>> nprod(lambda k: (1-1/k**2), [2, inf])
        0.5

    Next, several more infinite products with more complicated
    values::

        >>> nprod(lambda k: exp(1/k**2), [1, inf]); exp(pi**2/6)
        5.180668317897115748416626
        5.180668317897115748416626

        >>> nprod(lambda k: (k**2-1)/(k**2+1), [2, inf]); pi*csch(pi)
        0.2720290549821331629502366
        0.2720290549821331629502366

        >>> nprod(lambda k: (k**4-1)/(k**4+1), [2, inf])
        0.8480540493529003921296502
        >>> pi*sinh(pi)/(cosh(sqrt(2)*pi)-cos(sqrt(2)*pi))
        0.8480540493529003921296502

        >>> nprod(lambda k: (1+1/k+1/k**2)**2/(1+2/k+3/k**2), [1, inf])
        1.848936182858244485224927
        >>> 3*sqrt(2)*cosh(pi*sqrt(3)/2)**2*csch(pi*sqrt(2))/pi
        1.848936182858244485224927

        >>> nprod(lambda k: (1-1/k**4), [2, inf]); sinh(pi)/(4*pi)
        0.9190194775937444301739244
        0.9190194775937444301739244

        >>> nprod(lambda k: (1-1/k**6), [2, inf])
        0.9826842777421925183244759
        >>> (1+cosh(pi*sqrt(3)))/(12*pi**2)
        0.9826842777421925183244759

        >>> nprod(lambda k: (1+1/k**2), [2, inf]); sinh(pi)/(2*pi)
        1.838038955187488860347849
        1.838038955187488860347849

        >>> nprod(lambda n: (1+1/n)**n * exp(1/(2*n)-1), [1, inf])
        1.447255926890365298959138
        >>> exp(1+euler/2)/sqrt(2*pi)
        1.447255926890365298959138

    The following two products are equivalent and can be evaluated in
    terms of a Jacobi theta function. Pi can be replaced by any value
    (as long as convergence is preserved)::

        >>> nprod(lambda k: (1-pi**-k)/(1+pi**-k), [1, inf])
        0.3838451207481672404778686
        >>> nprod(lambda k: tanh(k*log(pi)/2), [1, inf])
        0.3838451207481672404778686
        >>> jtheta(4,0,1/pi)
        0.3838451207481672404778686

    This product does not have a known closed form value::

        >>> nprod(lambda k: (1-1/2**k), [1, inf])
        0.2887880950866024212788997

    A product taken from `-\\infty`::

        >>> nprod(lambda k: 1-k**(-3), [-inf,-2])
        0.8093965973662901095786805
        >>> cosh(pi*sqrt(3)/2)/(3*pi)
        0.8093965973662901095786805

    A doubly infinite product::

        >>> nprod(lambda k: exp(1/(1+k**2)), [-inf, inf])
        23.41432688231864337420035
        >>> exp(pi/tanh(pi))
        23.41432688231864337420035

    A product requiring the use of Euler-Maclaurin summation to compute
    an accurate value::

        >>> nprod(lambda k: (1-1/k**2.5), [2, inf], method='e')
        0.696155111336231052898125

    **References**

    1. [Weisstein]_ http://mathworld.wolfram.com/InfiniteProduct.html

    """
    if nsum or 'e' in kwargs.get('method', ''):
        orig = ctx.prec
        try:
            ctx.prec += 10
            v = ctx.nsum(lambda n: ctx.ln(f(n)), interval, **kwargs)
        finally:
            ctx.prec = orig
        return +ctx.exp(v)
    a, b = ctx._as_points(interval)
    if a == ctx.ninf:
        if b == ctx.inf:
            return f(0) * ctx.nprod(lambda k: f(-k) * f(k), [1, ctx.inf], **kwargs)
        return ctx.nprod(f, [-b, ctx.inf], **kwargs)
    elif b != ctx.inf:
        return ctx.fprod((f(ctx.mpf(k)) for k in xrange(int(a), int(b) + 1)))
    a = int(a)

    def update(partial_products, indices):
        if partial_products:
            pprod = partial_products[-1]
        else:
            pprod = ctx.one
        for k in indices:
            pprod = pprod * f(a + ctx.mpf(k))
            partial_products.append(pprod)
    return +ctx.adaptive_extrapolation(update, None, kwargs)