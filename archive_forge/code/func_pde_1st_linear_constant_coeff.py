from functools import reduce
from itertools import combinations_with_replacement
from sympy.simplify import simplify  # type: ignore
from sympy.core import Add, S
from sympy.core.function import Function, expand, AppliedUndef, Subs
from sympy.core.relational import Equality, Eq
from sympy.core.symbol import Symbol, Wild, symbols
from sympy.functions import exp
from sympy.integrals.integrals import Integral, integrate
from sympy.utilities.iterables import has_dups, is_sequence
from sympy.utilities.misc import filldedent
from sympy.solvers.deutils import _preprocess, ode_order, _desolve
from sympy.solvers.solvers import solve
from sympy.simplify.radsimp import collect
import operator
def pde_1st_linear_constant_coeff(eq, func, order, match, solvefun):
    """
    Solves a first order linear partial differential equation
    with constant coefficients.

    The general form of this partial differential equation is

    .. math:: a \\frac{\\partial f(x,y)}{\\partial x}
              + b \\frac{\\partial f(x,y)}{\\partial y}
              + c f(x,y) = G(x,y)

    where `a`, `b` and `c` are constants and `G(x, y)` can be an arbitrary
    function in `x` and `y`.

    The general solution of the PDE is:

    .. math::
        f(x, y) = \\left. \\left[F(\\eta) + \\frac{1}{a^2 + b^2}
        \\int\\limits^{a x + b y} G\\left(\\frac{a \\xi + b \\eta}{a^2 + b^2},
        \\frac{- a \\eta + b \\xi}{a^2 + b^2} \\right)
        e^{\\frac{c \\xi}{a^2 + b^2}}\\, d\\xi\\right]
        e^{- \\frac{c \\xi}{a^2 + b^2}}
        \\right|_{\\substack{\\eta=- a y + b x\\\\ \\xi=a x + b y }}\\, ,

    where `F(\\eta)` is an arbitrary single-valued function. The solution
    can be found in SymPy with ``pdsolve``::

        >>> from sympy.solvers import pdsolve
        >>> from sympy.abc import x, y, a, b, c
        >>> from sympy import Function, pprint
        >>> f = Function('f')
        >>> G = Function('G')
        >>> u = f(x,y)
        >>> ux = u.diff(x)
        >>> uy = u.diff(y)
        >>> genform = a*ux + b*uy + c*u - G(x,y)
        >>> pprint(genform)
          d               d
        a*--(f(x, y)) + b*--(f(x, y)) + c*f(x, y) - G(x, y)
          dx              dy
        >>> pprint(pdsolve(genform, hint='1st_linear_constant_coeff_Integral'))
                  //          a*x + b*y                                             \\
                  ||              /                                                 |
                  ||             |                                                  |
                  ||             |                                       c*xi       |
                  ||             |                                     -------      |
                  ||             |                                      2    2      |
                  ||             |      /a*xi + b*eta  -a*eta + b*xi\\  a  + b       |
                  ||             |     G|------------, -------------|*e        d(xi)|
                  ||             |      |   2    2         2    2   |               |
                  ||             |      \\  a  + b         a  + b    /               |
                  ||             |                                                  |
                  ||            /                                                   |
                  ||                                                                |
        f(x, y) = ||F(eta) + -------------------------------------------------------|*
                  ||                                  2    2                        |
                  \\\\                                 a  + b                         /
        <BLANKLINE>
                \\|
                ||
                ||
                ||
                ||
                ||
                ||
                ||
                ||
          -c*xi ||
         -------||
          2    2||
         a  + b ||
        e       ||
                ||
                /|eta=-a*y + b*x, xi=a*x + b*y


    Examples
    ========

    >>> from sympy.solvers.pde import pdsolve
    >>> from sympy import Function, pprint, exp
    >>> from sympy.abc import x,y
    >>> f = Function('f')
    >>> eq = -2*f(x,y).diff(x) + 4*f(x,y).diff(y) + 5*f(x,y) - exp(x + 3*y)
    >>> pdsolve(eq)
    Eq(f(x, y), (F(4*x + 2*y)*exp(x/2) + exp(x + 4*y)/15)*exp(-y))

    References
    ==========

    - Viktor Grigoryan, "Partial Differential Equations"
      Math 124A - Fall 2010, pp.7

    """
    xi, eta = symbols('xi eta')
    f = func.func
    x = func.args[0]
    y = func.args[1]
    b = match[match['b']]
    c = match[match['c']]
    d = match[match['d']]
    e = -match[match['e']]
    expterm = exp(-S(d) / (b ** 2 + c ** 2) * xi)
    functerm = solvefun(eta)
    solvedict = solve((b * x + c * y - xi, c * x - b * y - eta), x, y)
    genterm = 1 / S(b ** 2 + c ** 2) * Integral((1 / expterm * e).subs(solvedict), (xi, b * x + c * y))
    return Eq(f(x, y), Subs(expterm * (functerm + genterm), (eta, xi), (c * x - b * y, b * x + c * y)))