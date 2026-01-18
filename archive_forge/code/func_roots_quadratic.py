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
def roots_quadratic(f):
    """Returns a list of roots of a quadratic polynomial. If the domain is ZZ
    then the roots will be sorted with negatives coming before positives.
    The ordering will be the same for any numerical coefficients as long as
    the assumptions tested are correct, otherwise the ordering will not be
    sorted (but will be canonical).
    """
    a, b, c = f.all_coeffs()
    dom = f.get_domain()

    def _sqrt(d):
        co = []
        other = []
        for di in Mul.make_args(d):
            if di.is_Pow and di.exp.is_Integer and (di.exp % 2 == 0):
                co.append(Pow(di.base, di.exp // 2))
            else:
                other.append(di)
        if co:
            d = Mul(*other)
            co = Mul(*co)
            return co * sqrt(d)
        return sqrt(d)

    def _simplify(expr):
        if dom.is_Composite:
            return factor(expr)
        else:
            from sympy.simplify.simplify import simplify
            return simplify(expr)
    if c is S.Zero:
        r0, r1 = (S.Zero, -b / a)
        if not dom.is_Numerical:
            r1 = _simplify(r1)
        elif r1.is_negative:
            r0, r1 = (r1, r0)
    elif b is S.Zero:
        r = -c / a
        if not dom.is_Numerical:
            r = _simplify(r)
        R = _sqrt(r)
        r0 = -R
        r1 = R
    else:
        d = b ** 2 - 4 * a * c
        A = 2 * a
        B = -b / A
        if not dom.is_Numerical:
            d = _simplify(d)
            B = _simplify(B)
        D = factor_terms(_sqrt(d) / A)
        r0 = B - D
        r1 = B + D
        if a.is_negative:
            r0, r1 = (r1, r0)
        elif not dom.is_Numerical:
            r0, r1 = [expand_2arg(i) for i in (r0, r1)]
    return [r0, r1]