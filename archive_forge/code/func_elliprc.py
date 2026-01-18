from .functions import defun, defun_wrapped
@defun
def elliprc(ctx, x, y, pv=True):
    """
    Evaluates the degenerate Carlson symmetric elliptic integral
    of the first kind

    .. math ::

        R_C(x,y) = R_F(x,y,y) =
            \\frac{1}{2} \\int_0^{\\infty} \\frac{dt}{(t+y) \\sqrt{(t+x)}}.

    If `y \\in (-\\infty,0)`, either a value defined by continuity,
    or with *pv=True* the Cauchy principal value, can be computed.

    If `x \\ge 0, y > 0`, the value can be expressed in terms of
    elementary functions as

    .. math ::

        R_C(x,y) =
        \\begin{cases}
          \\dfrac{1}{\\sqrt{y-x}}
            \\cos^{-1}\\left(\\sqrt{\\dfrac{x}{y}}\\right),   & x < y \\\\
          \\dfrac{1}{\\sqrt{y}},                          & x = y \\\\
          \\dfrac{1}{\\sqrt{x-y}}
            \\cosh^{-1}\\left(\\sqrt{\\dfrac{x}{y}}\\right),  & x > y \\\\
        \\end{cases}.

    **Examples**

    Some special values and limits::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> elliprc(1,2)*4; elliprc(0,1)*2; +pi
        3.141592653589793238462643
        3.141592653589793238462643
        3.141592653589793238462643
        >>> elliprc(1,0)
        +inf
        >>> elliprc(5,5)**2
        0.2
        >>> elliprc(1,inf); elliprc(inf,1); elliprc(inf,inf)
        0.0
        0.0
        0.0

    Comparing with the elementary closed-form solution::

        >>> elliprc('1/3', '1/5'); sqrt(7.5)*acosh(sqrt('5/3'))
        2.041630778983498390751238
        2.041630778983498390751238
        >>> elliprc('1/5', '1/3'); sqrt(7.5)*acos(sqrt('3/5'))
        1.875180765206547065111085
        1.875180765206547065111085

    Comparing with numerical integration::

        >>> q = extradps(25)(quad)
        >>> elliprc(2, -3, pv=True)
        0.3333969101113672670749334
        >>> elliprc(2, -3, pv=False)
        (0.3333969101113672670749334 + 0.7024814731040726393156375j)
        >>> 0.5*q(lambda t: 1/(sqrt(t+2)*(t-3)), [0,3-j,6,inf])
        (0.3333969101113672670749334 + 0.7024814731040726393156375j)

    """
    x = ctx.convert(x)
    y = ctx.convert(y)
    prec = ctx.prec
    try:
        ctx.prec += 20
        tol = ctx.eps * 2 ** 10
        v = RC_calc(ctx, x, y, tol, pv)
    finally:
        ctx.prec = prec
    return +v