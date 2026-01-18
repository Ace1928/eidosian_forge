from .functions import defun, defun_wrapped
@defun_wrapped
def kleinj(ctx, tau=None, **kwargs):
    """
    Evaluates the Klein j-invariant, which is a modular function defined for
    `\\tau` in the upper half-plane as

    .. math ::

        J(\\tau) = \\frac{g_2^3(\\tau)}{g_2^3(\\tau) - 27 g_3^2(\\tau)}

    where `g_2` and `g_3` are the modular invariants of the Weierstrass
    elliptic function,

    .. math ::

        g_2(\\tau) = 60 \\sum_{(m,n) \\in \\mathbb{Z}^2 \\setminus (0,0)} (m \\tau+n)^{-4}

        g_3(\\tau) = 140 \\sum_{(m,n) \\in \\mathbb{Z}^2 \\setminus (0,0)} (m \\tau+n)^{-6}.

    An alternative, common notation is that of the j-function
    `j(\\tau) = 1728 J(\\tau)`.

    **Plots**

    .. literalinclude :: /plots/kleinj.py
    .. image :: /plots/kleinj.png
    .. literalinclude :: /plots/kleinj2.py
    .. image :: /plots/kleinj2.png

    **Examples**

    Verifying the functional equation `J(\\tau) = J(\\tau+1) = J(-\\tau^{-1})`::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> tau = 0.625+0.75*j
        >>> tau = 0.625+0.75*j
        >>> kleinj(tau)
        (-0.1507492166511182267125242 + 0.07595948379084571927228948j)
        >>> kleinj(tau+1)
        (-0.1507492166511182267125242 + 0.07595948379084571927228948j)
        >>> kleinj(-1/tau)
        (-0.1507492166511182267125242 + 0.07595948379084571927228946j)

    The j-function has a famous Laurent series expansion in terms of the nome
    `\\bar{q}`, `j(\\tau) = \\bar{q}^{-1} + 744 + 196884\\bar{q} + \\ldots`::

        >>> mp.dps = 15
        >>> taylor(lambda q: 1728*q*kleinj(qbar=q), 0, 5, singular=True)
        [1.0, 744.0, 196884.0, 21493760.0, 864299970.0, 20245856256.0]

    The j-function admits exact evaluation at special algebraic points
    related to the Heegner numbers 1, 2, 3, 7, 11, 19, 43, 67, 163::

        >>> @extraprec(10)
        ... def h(n):
        ...     v = (1+sqrt(n)*j)
        ...     if n > 2:
        ...         v *= 0.5
        ...     return v
        ...
        >>> mp.dps = 25
        >>> for n in [1,2,3,7,11,19,43,67,163]:
        ...     n, chop(1728*kleinj(h(n)))
        ...
        (1, 1728.0)
        (2, 8000.0)
        (3, 0.0)
        (7, -3375.0)
        (11, -32768.0)
        (19, -884736.0)
        (43, -884736000.0)
        (67, -147197952000.0)
        (163, -262537412640768000.0)

    Also at other special points, the j-function assumes explicit
    algebraic values, e.g.::

        >>> chop(1728*kleinj(j*sqrt(5)))
        1264538.909475140509320227
        >>> identify(cbrt(_))      # note: not simplified
        '((100+sqrt(13520))/2)'
        >>> (50+26*sqrt(5))**3
        1264538.909475140509320227

    """
    q = ctx.qfrom(tau=tau, **kwargs)
    t2 = ctx.jtheta(2, 0, q)
    t3 = ctx.jtheta(3, 0, q)
    t4 = ctx.jtheta(4, 0, q)
    P = (t2 ** 8 + t3 ** 8 + t4 ** 8) ** 3
    Q = 54 * (t2 * t3 * t4) ** 8
    return P / Q