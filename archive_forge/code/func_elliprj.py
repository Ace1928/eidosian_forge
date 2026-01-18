from .functions import defun, defun_wrapped
@defun
def elliprj(ctx, x, y, z, p, integration=1):
    """
    Evaluates the Carlson symmetric elliptic integral of the third kind

    .. math ::

        R_J(x,y,z,p) = \\frac{3}{2}
            \\int_0^{\\infty} \\frac{dt}{(t+p)\\sqrt{(t+x)(t+y)(t+z)}}.

    Like :func:`~mpmath.elliprf`, the branch of the square root in the integrand
    is defined so as to be continuous along the path of integration for
    complex values of the arguments.

    **Examples**

    Some values and limits::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> elliprj(1,1,1,1)
        1.0
        >>> elliprj(2,2,2,2); 1/(2*sqrt(2))
        0.3535533905932737622004222
        0.3535533905932737622004222
        >>> elliprj(0,1,2,2)
        1.067937989667395702268688
        >>> 3*(2*gamma('5/4')**2-pi**2/gamma('1/4')**2)/(sqrt(2*pi))
        1.067937989667395702268688
        >>> elliprj(0,1,1,2); 3*pi*(2-sqrt(2))/4
        1.380226776765915172432054
        1.380226776765915172432054
        >>> elliprj(1,3,2,0); elliprj(0,1,1,0); elliprj(0,0,0,0)
        +inf
        +inf
        +inf
        >>> elliprj(1,inf,1,0); elliprj(1,1,1,inf)
        0.0
        0.0
        >>> chop(elliprj(1+j, 1-j, 1, 1))
        0.8505007163686739432927844

    Scale transformation::

        >>> x,y,z,p = 2,3,4,5
        >>> k = mpf(100000)
        >>> elliprj(k*x,k*y,k*z,k*p); k**(-1.5)*elliprj(x,y,z,p)
        4.521291677592745527851168e-9
        4.521291677592745527851168e-9

    Comparing with numerical integration::

        >>> elliprj(1,2,3,4)
        0.2398480997495677621758617
        >>> f = lambda t: 1/((t+4)*sqrt((t+1)*(t+2)*(t+3)))
        >>> 1.5*quad(f, [0,inf])
        0.2398480997495677621758617
        >>> elliprj(1,2+1j,3,4-2j)
        (0.216888906014633498739952 + 0.04081912627366673332369512j)
        >>> f = lambda t: 1/((t+4-2j)*sqrt((t+1)*(t+2+1j)*(t+3)))
        >>> 1.5*quad(f, [0,inf])
        (0.216888906014633498739952 + 0.04081912627366673332369511j)

    """
    x = ctx.convert(x)
    y = ctx.convert(y)
    z = ctx.convert(z)
    p = ctx.convert(p)
    prec = ctx.prec
    try:
        ctx.prec += 20
        tol = ctx.eps * 2 ** 10
        v = RJ_calc(ctx, x, y, z, p, tol, integration)
    finally:
        ctx.prec = prec
    return +v