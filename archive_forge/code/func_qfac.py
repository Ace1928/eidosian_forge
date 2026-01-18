from .functions import defun, defun_wrapped
@defun_wrapped
def qfac(ctx, z, q, **kwargs):
    """
    Evaluates the q-factorial,

    .. math ::

        [n]_q! = (1+q)(1+q+q^2)\\cdots(1+q+\\cdots+q^{n-1})

    or more generally

    .. math ::

        [z]_q! = \\frac{(q;q)_z}{(1-q)^z}.

    **Examples**

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> qfac(0,0)
        1.0
        >>> qfac(4,3)
        2080.0
        >>> qfac(5,6)
        121226245.0
        >>> qfac(1+1j, 2+1j)
        (0.4370556551322672478613695 + 0.2609739839216039203708921j)

    """
    if ctx.isint(z) and ctx._re(z) > 0:
        n = int(ctx._re(z))
        return ctx.qp(q, q, n, **kwargs) / (1 - q) ** n
    return ctx.qgamma(z + 1, q, **kwargs)