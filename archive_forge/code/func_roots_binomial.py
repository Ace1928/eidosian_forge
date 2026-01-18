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
def roots_binomial(f):
    """Returns a list of roots of a binomial polynomial. If the domain is ZZ
    then the roots will be sorted with negatives coming before positives.
    The ordering will be the same for any numerical coefficients as long as
    the assumptions tested are correct, otherwise the ordering will not be
    sorted (but will be canonical).
    """
    n = f.degree()
    a, b = (f.nth(n), f.nth(0))
    base = -cancel(b / a)
    alpha = root(base, n)
    if alpha.is_number:
        alpha = alpha.expand(complex=True)
    neg = base.is_negative
    even = n % 2 == 0
    if neg:
        if even == True and (base + 1).is_positive:
            big = True
        else:
            big = False
    ks = []
    imax = n // 2
    if even:
        ks.append(imax)
        imax -= 1
    if not neg:
        ks.append(0)
    for i in range(imax, 0, -1):
        if neg:
            ks.extend([i, -i])
        else:
            ks.extend([-i, i])
    if neg:
        ks.append(0)
        if big:
            for i in range(0, len(ks), 2):
                pair = ks[i:i + 2]
                pair = list(reversed(pair))
    roots, d = ([], 2 * I * pi / n)
    for k in ks:
        zeta = exp(k * d).expand(complex=True)
        roots.append((alpha * zeta).expand(power_base=False))
    return roots