from .functions import defun, defun_wrapped
@defun
def elliprg(ctx, x, y, z):
    """
    Evaluates the Carlson completely symmetric elliptic integral
    of the second kind

    .. math ::

        R_G(x,y,z) = \\frac{1}{4} \\int_0^{\\infty}
            \\frac{t}{\\sqrt{(t+x)(t+y)(t+z)}}
            \\left( \\frac{x}{t+x} + \\frac{y}{t+y} + \\frac{z}{t+z}\\right) dt.

    **Examples**

    Evaluation for real and complex arguments::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> elliprg(0,1,1)*4; +pi
        3.141592653589793238462643
        3.141592653589793238462643
        >>> elliprg(0,0.5,1)
        0.6753219405238377512600874
        >>> chop(elliprg(1+j, 1-j, 2))
        1.172431327676416604532822

    A double integral that can be evaluated in terms of `R_G`::

        >>> x,y,z = 2,3,4
        >>> def f(t,u):
        ...     st = fp.sin(t); ct = fp.cos(t)
        ...     su = fp.sin(u); cu = fp.cos(u)
        ...     return (x*(st*cu)**2 + y*(st*su)**2 + z*ct**2)**0.5 * st
        ...
        >>> nprint(mpf(fp.quad(f, [0,fp.pi], [0,2*fp.pi])/(4*fp.pi)), 13)
        1.725503028069
        >>> nprint(elliprg(x,y,z), 13)
        1.725503028069

    """
    x = ctx.convert(x)
    y = ctx.convert(y)
    z = ctx.convert(z)
    zeros = (not x) + (not y) + (not z)
    if zeros == 3:
        return (x + y + z) * 0
    if zeros == 2:
        if x:
            return 0.5 * ctx.sqrt(x)
        if y:
            return 0.5 * ctx.sqrt(y)
        return 0.5 * ctx.sqrt(z)
    if zeros == 1:
        if not z:
            x, z = (z, x)

    def terms():
        T1 = 0.5 * z * ctx.elliprf(x, y, z)
        T2 = -0.5 * (x - z) * (y - z) * ctx.elliprd(x, y, z) / 3
        T3 = 0.5 * ctx.sqrt(x) * ctx.sqrt(y) / ctx.sqrt(z)
        return (T1, T2, T3)
    return ctx.sum_accurately(terms)