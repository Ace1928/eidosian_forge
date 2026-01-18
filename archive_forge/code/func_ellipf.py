from .functions import defun, defun_wrapped
@defun_wrapped
def ellipf(ctx, phi, m):
    """
    Evaluates the Legendre incomplete elliptic integral of the first kind

     .. math ::

        F(\\phi,m) = \\int_0^{\\phi} \\frac{dt}{\\sqrt{1-m \\sin^2 t}}

    or equivalently

    .. math ::

        F(\\phi,m) = \\int_0^{\\sin \\phi}
        \\frac{dt}{\\left(\\sqrt{1-t^2}\\right)\\left(\\sqrt{1-mt^2}\\right)}.

    The function reduces to a complete elliptic integral of the first kind
    (see :func:`~mpmath.ellipk`) when `\\phi = \\frac{\\pi}{2}`; that is,

    .. math ::

        F\\left(\\frac{\\pi}{2}, m\\right) = K(m).

    In the defining integral, it is assumed that the principal branch
    of the square root is taken and that the path of integration avoids
    crossing any branch cuts. Outside `-\\pi/2 \\le \\Re(\\phi) \\le \\pi/2`,
    the function extends quasi-periodically as

    .. math ::

        F(\\phi + n \\pi, m) = 2 n K(m) + F(\\phi,m), n \\in \\mathbb{Z}.

    **Plots**

    .. literalinclude :: /plots/ellipf.py
    .. image :: /plots/ellipf.png

    **Examples**

    Basic values and limits::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> ellipf(0,1)
        0.0
        >>> ellipf(0,0)
        0.0
        >>> ellipf(1,0); ellipf(2+3j,0)
        1.0
        (2.0 + 3.0j)
        >>> ellipf(1,1); log(sec(1)+tan(1))
        1.226191170883517070813061
        1.226191170883517070813061
        >>> ellipf(pi/2, -0.5); ellipk(-0.5)
        1.415737208425956198892166
        1.415737208425956198892166
        >>> ellipf(pi/2+eps, 1); ellipf(-pi/2-eps, 1)
        +inf
        +inf
        >>> ellipf(1.5, 1)
        3.340677542798311003320813

    Comparing with numerical integration::

        >>> z,m = 0.5, 1.25
        >>> ellipf(z,m)
        0.5287219202206327872978255
        >>> quad(lambda t: (1-m*sin(t)**2)**(-0.5), [0,z])
        0.5287219202206327872978255

    The arguments may be complex numbers::

        >>> ellipf(3j, 0.5)
        (0.0 + 1.713602407841590234804143j)
        >>> ellipf(3+4j, 5-6j)
        (1.269131241950351323305741 - 0.3561052815014558335412538j)
        >>> z,m = 2+3j, 1.25
        >>> k = 1011
        >>> ellipf(z+pi*k,m); ellipf(z,m) + 2*k*ellipk(m)
        (4086.184383622179764082821 - 3003.003538923749396546871j)
        (4086.184383622179764082821 - 3003.003538923749396546871j)

    For `|\\Re(z)| < \\pi/2`, the function can be expressed as a
    hypergeometric series of two variables
    (see :func:`~mpmath.appellf1`)::

        >>> z,m = 0.5, 0.25
        >>> ellipf(z,m)
        0.5050887275786480788831083
        >>> sin(z)*appellf1(0.5,0.5,0.5,1.5,sin(z)**2,m*sin(z)**2)
        0.5050887275786480788831083

    """
    z = phi
    if not (ctx.isnormal(z) and ctx.isnormal(m)):
        if m == 0:
            return z + m
        if z == 0:
            return z * m
        if m == ctx.inf or m == ctx.ninf:
            return z / m
        raise ValueError
    x = z.real
    ctx.prec += max(0, ctx.mag(x))
    pi = +ctx.pi
    away = abs(x) > pi / 2
    if m == 1:
        if away:
            return ctx.inf
    if away:
        d = ctx.nint(x / pi)
        z = z - pi * d
        P = 2 * d * ctx.ellipk(m)
    else:
        P = 0
    c, s = ctx.cos_sin(z)
    return s * ctx.elliprf(c ** 2, 1 - m * s ** 2, 1) + P