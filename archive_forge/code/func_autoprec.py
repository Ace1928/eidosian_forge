import functools
import re
from .ctx_base import StandardBaseContext
from .libmp.backend import basestring, BACKEND
from . import libmp
from .libmp import (MPZ, MPZ_ZERO, MPZ_ONE, int_types, repr_dps,
from . import function_docs
from . import rational
from .ctx_mp_python import _mpf, _mpc, mpnumeric
def autoprec(ctx, f, maxprec=None, catch=(), verbose=False):
    """
        Return a wrapped copy of *f* that repeatedly evaluates *f*
        with increasing precision until the result converges to the
        full precision used at the point of the call.

        This heuristically protects against rounding errors, at the cost of
        roughly a 2x slowdown compared to manually setting the optimal
        precision. This method can, however, easily be fooled if the results
        from *f* depend "discontinuously" on the precision, for instance
        if catastrophic cancellation can occur. Therefore, :func:`~mpmath.autoprec`
        should be used judiciously.

        **Examples**

        Many functions are sensitive to perturbations of the input arguments.
        If the arguments are decimal numbers, they may have to be converted
        to binary at a much higher precision. If the amount of required
        extra precision is unknown, :func:`~mpmath.autoprec` is convenient::

            >>> from mpmath import *
            >>> mp.dps = 15
            >>> mp.pretty = True
            >>> besselj(5, 125 * 10**28)    # Exact input
            -8.03284785591801e-17
            >>> besselj(5, '1.25e30')   # Bad
            7.12954868316652e-16
            >>> autoprec(besselj)(5, '1.25e30')   # Good
            -8.03284785591801e-17

        The following fails to converge because `\\sin(\\pi) = 0` whereas all
        finite-precision approximations of `\\pi` give nonzero values::

            >>> autoprec(sin)(pi) # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
              ...
            NoConvergence: autoprec: prec increased to 2910 without convergence

        As the following example shows, :func:`~mpmath.autoprec` can protect against
        cancellation, but is fooled by too severe cancellation::

            >>> x = 1e-10
            >>> exp(x)-1; expm1(x); autoprec(lambda t: exp(t)-1)(x)
            1.00000008274037e-10
            1.00000000005e-10
            1.00000000005e-10
            >>> x = 1e-50
            >>> exp(x)-1; expm1(x); autoprec(lambda t: exp(t)-1)(x)
            0.0
            1.0e-50
            0.0

        With *catch*, an exception or list of exceptions to intercept
        may be specified. The raised exception is interpreted
        as signaling insufficient precision. This permits, for example,
        evaluating a function where a too low precision results in a
        division by zero::

            >>> f = lambda x: 1/(exp(x)-1)
            >>> f(1e-30)
            Traceback (most recent call last):
              ...
            ZeroDivisionError
            >>> autoprec(f, catch=ZeroDivisionError)(1e-30)
            1.0e+30


        """

    def f_autoprec_wrapped(*args, **kwargs):
        prec = ctx.prec
        if maxprec is None:
            maxprec2 = ctx._default_hyper_maxprec(prec)
        else:
            maxprec2 = maxprec
        try:
            ctx.prec = prec + 10
            try:
                v1 = f(*args, **kwargs)
            except catch:
                v1 = ctx.nan
            prec2 = prec + 20
            while 1:
                ctx.prec = prec2
                try:
                    v2 = f(*args, **kwargs)
                except catch:
                    v2 = ctx.nan
                if v1 == v2:
                    break
                err = ctx.mag(v2 - v1) - ctx.mag(v2)
                if err < -prec:
                    break
                if verbose:
                    print('autoprec: target=%s, prec=%s, accuracy=%s' % (prec, prec2, -err))
                v1 = v2
                if prec2 >= maxprec2:
                    raise ctx.NoConvergence('autoprec: prec increased to %i without convergence' % prec2)
                prec2 += int(prec2 * 2)
                prec2 = min(prec2, maxprec2)
        finally:
            ctx.prec = prec
        return +v2
    return f_autoprec_wrapped