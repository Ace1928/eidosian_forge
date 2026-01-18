from .functions import defun, defun_wrapped
@defun_wrapped
def qgamma(ctx, z, q, **kwargs):
    """
    Evaluates the q-gamma function

    .. math ::

        \\Gamma_q(z) = \\frac{(q; q)_{\\infty}}{(q^z; q)_{\\infty}} (1-q)^{1-z}.


    **Examples**

    Evaluation for real and complex arguments::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> qgamma(4,0.75)
        4.046875
        >>> qgamma(6,6)
        121226245.0
        >>> qgamma(3+4j, 0.5j)
        (0.1663082382255199834630088 + 0.01952474576025952984418217j)

    The q-gamma function satisfies a functional equation similar
    to that of the ordinary gamma function::

        >>> q = mpf(0.25)
        >>> z = mpf(2.5)
        >>> qgamma(z+1,q)
        1.428277424823760954685912
        >>> (1-q**z)/(1-q)*qgamma(z,q)
        1.428277424823760954685912

    """
    if abs(q) > 1:
        return ctx.qgamma(z, 1 / q) * q ** ((z - 2) * (z - 1) * 0.5)
    return ctx.qp(q, q, None, **kwargs) / ctx.qp(q ** z, q, None, **kwargs) * (1 - q) ** (1 - z)