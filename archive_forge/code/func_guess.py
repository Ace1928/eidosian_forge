from sympy.concrete.products import (Product, product)
from sympy.core import Function, S
from sympy.core.add import Add
from sympy.core.numbers import Integer, Rational
from sympy.core.symbol import Symbol, symbols
from sympy.core.sympify import sympify
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.integers import floor
from sympy.integrals.integrals import integrate
from sympy.polys.polyfuncs import rational_interpolate as rinterp
from sympy.polys.polytools import lcm
from sympy.simplify.radsimp import denom
from sympy.utilities import public
@public
def guess(l, all=False, evaluate=True, niter=2, variables=None):
    """
    This function is adapted from the Rate.m package for Mathematica
    written by Christian Krattenthaler.
    It tries to guess a formula from a given sequence of rational numbers.

    Explanation
    ===========

    In order to speed up the process, the 'all' variable is set to False by
    default, stopping the computation as some results are returned during an
    iteration; the variable can be set to True if more iterations are needed
    (other formulas may be found; however they may be equivalent to the first
    ones).

    Another option is the 'evaluate' variable (default is True); setting it
    to False will leave the involved products unevaluated.

    By default, the number of iterations is set to 2 but a greater value (up
    to len(l)-1) can be specified with the optional 'niter' variable.
    More and more convoluted results are found when the order of the
    iteration gets higher:

      * first iteration returns polynomial or rational functions;
      * second iteration returns products of rising factorials and their
        inverses;
      * third iteration returns products of products of rising factorials
        and their inverses;
      * etc.

    The returned formulas contain symbols i0, i1, i2, ... where the main
    variables is i0 (and auxiliary variables are i1, i2, ...). A list of
    other symbols can be provided in the 'variables' option; the length of
    the least should be the value of 'niter' (more is acceptable but only
    the first symbols will be used); in this case, the main variable will be
    the first symbol in the list.

    Examples
    ========

    >>> from sympy.concrete.guess import guess
    >>> guess([1,2,6,24,120], evaluate=False)
    [Product(i1 + 1, (i1, 1, i0 - 1))]

    >>> from sympy import symbols
    >>> r = guess([1,2,7,42,429,7436,218348,10850216], niter=4)
    >>> i0 = symbols("i0")
    >>> [r[0].subs(i0,n).doit() for n in range(1,10)]
    [1, 2, 7, 42, 429, 7436, 218348, 10850216, 911835460]
    """
    if any((a == 0 for a in l[:-1])):
        return []
    N = len(l)
    niter = min(N - 1, niter)
    myprod = product if evaluate else Product
    g = []
    res = []
    if variables is None:
        symb = symbols('i:' + str(niter))
    else:
        symb = variables
    for k, s in enumerate(symb):
        g.append(l)
        n, r = (len(l), [])
        for i in range(n - 2 - 1, -1, -1):
            ri = rinterp(enumerate(g[k][:-1], start=1), i, X=s)
            if denom(ri).subs({s: n}) != 0 and ri.subs({s: n}) - g[k][-1] == 0 and (ri not in r):
                r.append(ri)
        if r:
            for i in range(k - 1, -1, -1):
                r = [g[i][0] * myprod(v, (symb[i + 1], 1, symb[i] - 1)) for v in r]
            if not all:
                return r
            res += r
        l = [Rational(l[i + 1], l[i]) for i in range(N - k - 1)]
    return res