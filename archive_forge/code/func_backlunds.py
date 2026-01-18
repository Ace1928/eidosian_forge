from .functions import defun, defun_wrapped
@defun_wrapped
def backlunds(ctx, t):
    """
    Computes the function
    `S(t) = \\operatorname{arg} \\zeta(\\frac{1}{2} + it) / \\pi`.

    See Titchmarsh Section 9.3 for details of the definition.

    **Examples**

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> backlunds(217.3)
        0.16302205431184

    Generally, the value is a small number. At Gram points it is an integer,
    frequently equal to 0::

        >>> chop(backlunds(grampoint(200)))
        0.0
        >>> backlunds(extraprec(10)(grampoint)(211))
        1.0
        >>> backlunds(extraprec(10)(grampoint)(232))
        -1.0

    The number of zeros of the Riemann zeta function up to height `t`
    satisfies `N(t) = \\theta(t)/\\pi + 1 + S(t)` (see :func:nzeros` and
    :func:`siegeltheta`)::

        >>> t = 1234.55
        >>> nzeros(t)
        842
        >>> siegeltheta(t)/pi+1+backlunds(t)
        842.0

    """
    return ctx.nzeros(t) - 1 - ctx.siegeltheta(t) / ctx.pi