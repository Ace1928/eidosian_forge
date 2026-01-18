from ..libmp.backend import xrange
from .functions import defun, defun_wrapped
@defun
def bihyper(ctx, a_s, b_s, z, **kwargs):
    """
    Evaluates the bilateral hypergeometric series

    .. math ::

        \\,_AH_B(a_1, \\ldots, a_k; b_1, \\ldots, b_B; z) =
            \\sum_{n=-\\infty}^{\\infty}
            \\frac{(a_1)_n \\ldots (a_A)_n}
                 {(b_1)_n \\ldots (b_B)_n} \\, z^n

    where, for direct convergence, `A = B` and `|z| = 1`, although a
    regularized sum exists more generally by considering the
    bilateral series as a sum of two ordinary hypergeometric
    functions. In order for the series to make sense, none of the
    parameters may be integers.

    **Examples**

    The value of `\\,_2H_2` at `z = 1` is given by Dougall's formula::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> a,b,c,d = 0.5, 1.5, 2.25, 3.25
        >>> bihyper([a,b],[c,d],1)
        -14.49118026212345786148847
        >>> gammaprod([c,d,1-a,1-b,c+d-a-b-1],[c-a,d-a,c-b,d-b])
        -14.49118026212345786148847

    The regularized function `\\,_1H_0` can be expressed as the
    sum of one `\\,_2F_0` function and one `\\,_1F_1` function::

        >>> a = mpf(0.25)
        >>> z = mpf(0.75)
        >>> bihyper([a], [], z)
        (0.2454393389657273841385582 + 0.2454393389657273841385582j)
        >>> hyper([a,1],[],z) + (hyper([1],[1-a],-1/z)-1)
        (0.2454393389657273841385582 + 0.2454393389657273841385582j)
        >>> hyper([a,1],[],z) + hyper([1],[2-a],-1/z)/z/(a-1)
        (0.2454393389657273841385582 + 0.2454393389657273841385582j)

    **References**

    1. [Slater]_ (chapter 6: "Bilateral Series", pp. 180-189)
    2. [Wikipedia]_ http://en.wikipedia.org/wiki/Bilateral_hypergeometric_series

    """
    z = ctx.convert(z)
    c_s = a_s + b_s
    p = len(a_s)
    q = len(b_s)
    if (p, q) == (0, 0) or (p, q) == (1, 1):
        return ctx.zero * z
    neg = (p - q) % 2

    def h(*c_s):
        a_s = list(c_s[:p])
        b_s = list(c_s[p:])
        aa_s = [2 - b for b in b_s]
        bb_s = [2 - a for a in a_s]
        rp = [(-1) ** neg * z] + [1 - b for b in b_s] + [1 - a for a in a_s]
        rc = [-1] + [1] * len(b_s) + [-1] * len(a_s)
        T1 = ([], [], [], [], a_s + [1], b_s, z)
        T2 = (rp, rc, [], [], aa_s + [1], bb_s, (-1) ** neg / z)
        return (T1, T2)
    return ctx.hypercomb(h, c_s, **kwargs)