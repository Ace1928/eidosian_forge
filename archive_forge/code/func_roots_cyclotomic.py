import math
from functools import reduce
from sympy.core import S, I, pi
from sympy.core.exprtools import factor_terms
from sympy.core.function import _mexpand
from sympy.core.logic import fuzzy_not
from sympy.core.mul import expand_2arg, Mul
from sympy.core.numbers import Rational, igcd, comp
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, Symbol, symbols
from sympy.core.sympify import sympify
from sympy.functions import exp, im, cos, acos, Piecewise
from sympy.functions.elementary.miscellaneous import root, sqrt
from sympy.ntheory import divisors, isprime, nextprime
from sympy.polys.domains import EX
from sympy.polys.polyerrors import (PolynomialError, GeneratorsNeeded,
from sympy.polys.polyquinticconst import PolyQuintic
from sympy.polys.polytools import Poly, cancel, factor, gcd_list, discriminant
from sympy.polys.rationaltools import together
from sympy.polys.specialpolys import cyclotomic_poly
from sympy.utilities import public
from sympy.utilities.misc import filldedent
def roots_cyclotomic(f, factor=False):
    """Compute roots of cyclotomic polynomials. """
    L, U = _inv_totient_estimate(f.degree())
    for n in range(L, U + 1):
        g = cyclotomic_poly(n, f.gen, polys=True)
        if f.expr == g.expr:
            break
    else:
        raise RuntimeError('failed to find index of a cyclotomic polynomial')
    roots = []
    if not factor:
        h = n // 2
        ks = [i for i in range(1, n + 1) if igcd(i, n) == 1]
        ks.sort(key=lambda x: (x, -1) if x <= h else (abs(x - n), 1))
        d = 2 * I * pi / n
        for k in reversed(ks):
            roots.append(exp(k * d).expand(complex=True))
    else:
        g = Poly(f, extension=root(-1, n))
        for h, _ in ordered(g.factor_list()[1]):
            roots.append(-h.TC())
    return roots