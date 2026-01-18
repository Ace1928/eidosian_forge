from ..libmp.backend import xrange
from .calculus import defun
@defun
def sumem(ctx, f, interval, tol=None, reject=10, integral=None, adiffs=None, bdiffs=None, verbose=False, error=False, _fast_abort=False):
    """
    Uses the Euler-Maclaurin formula to compute an approximation accurate
    to within ``tol`` (which defaults to the present epsilon) of the sum

    .. math ::

        S = \\sum_{k=a}^b f(k)

    where `(a,b)` are given by ``interval`` and `a` or `b` may be
    infinite. The approximation is

    .. math ::

        S \\sim \\int_a^b f(x) \\,dx + \\frac{f(a)+f(b)}{2} +
        \\sum_{k=1}^{\\infty} \\frac{B_{2k}}{(2k)!}
        \\left(f^{(2k-1)}(b)-f^{(2k-1)}(a)\\right).

    The last sum in the Euler-Maclaurin formula is not generally
    convergent (a notable exception is if `f` is a polynomial, in
    which case Euler-Maclaurin actually gives an exact result).

    The summation is stopped as soon as the quotient between two
    consecutive terms falls below *reject*. That is, by default
    (*reject* = 10), the summation is continued as long as each
    term adds at least one decimal.

    Although not convergent, convergence to a given tolerance can
    often be "forced" if `b = \\infty` by summing up to `a+N` and then
    applying the Euler-Maclaurin formula to the sum over the range
    `(a+N+1, \\ldots, \\infty)`. This procedure is implemented by
    :func:`~mpmath.nsum`.

    By default numerical quadrature and differentiation is used.
    If the symbolic values of the integral and endpoint derivatives
    are known, it is more efficient to pass the value of the
    integral explicitly as ``integral`` and the derivatives
    explicitly as ``adiffs`` and ``bdiffs``. The derivatives
    should be given as iterables that yield
    `f(a), f'(a), f''(a), \\ldots` (and the equivalent for `b`).

    **Examples**

    Summation of an infinite series, with automatic and symbolic
    integral and derivative values (the second should be much faster)::

        >>> from mpmath import *
        >>> mp.dps = 50; mp.pretty = True
        >>> sumem(lambda n: 1/n**2, [32, inf])
        0.03174336652030209012658168043874142714132886413417
        >>> I = mpf(1)/32
        >>> D = adiffs=((-1)**n*fac(n+1)*32**(-2-n) for n in range(999))
        >>> sumem(lambda n: 1/n**2, [32, inf], integral=I, adiffs=D)
        0.03174336652030209012658168043874142714132886413417

    An exact evaluation of a finite polynomial sum::

        >>> sumem(lambda n: n**5-12*n**2+3*n, [-100000, 200000])
        10500155000624963999742499550000.0
        >>> print(sum(n**5-12*n**2+3*n for n in range(-100000, 200001)))
        10500155000624963999742499550000

    """
    tol = tol or +ctx.eps
    interval = ctx._as_points(interval)
    a = ctx.convert(interval[0])
    b = ctx.convert(interval[-1])
    err = ctx.zero
    prev = 0
    M = 10000
    if a == ctx.ninf:
        adiffs = (0 for n in xrange(M))
    else:
        adiffs = adiffs or ctx.diffs(f, a)
    if b == ctx.inf:
        bdiffs = (0 for n in xrange(M))
    else:
        bdiffs = bdiffs or ctx.diffs(f, b)
    orig = ctx.prec
    try:
        ctx.prec += 10
        s = ctx.zero
        for k, (da, db) in enumerate(izip(adiffs, bdiffs)):
            if k & 1:
                term = (db - da) * ctx.bernoulli(k + 1) / ctx.factorial(k + 1)
                mag = abs(term)
                if verbose:
                    print('term', k, 'magnitude =', ctx.nstr(mag))
                if k > 4 and mag < tol:
                    s += term
                    break
                elif k > 4 and abs(prev) / mag < reject:
                    err += mag
                    if _fast_abort:
                        return [s, (s, err)][error]
                    if verbose:
                        print('Failed to converge')
                    break
                else:
                    s += term
                prev = term
        if a != ctx.ninf:
            s += f(a) / 2
        if b != ctx.inf:
            s += f(b) / 2
        if verbose:
            print('Integrating f(x) from x = %s to %s' % (ctx.nstr(a), ctx.nstr(b)))
        if integral:
            s += integral
        else:
            integral, ierr = ctx.quad(f, interval, error=True)
            if verbose:
                print('Integration error:', ierr)
            s += integral
            err += ierr
    finally:
        ctx.prec = orig
    if error:
        return (s, err)
    else:
        return s