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
def preprocess_roots(poly):
    """Try to get rid of symbolic coefficients from ``poly``. """
    coeff = S.One
    poly_func = poly.func
    try:
        _, poly = poly.clear_denoms(convert=True)
    except DomainError:
        return (coeff, poly)
    poly = poly.primitive()[1]
    poly = poly.retract()
    if poly.get_domain().is_Poly and all((c.is_term for c in poly.rep.coeffs())):
        poly = poly.inject()
        strips = list(zip(*poly.monoms()))
        gens = list(poly.gens[1:])
        base, strips = (strips[0], strips[1:])
        for gen, strip in zip(list(gens), strips):
            reverse = False
            if strip[0] < strip[-1]:
                strip = reversed(strip)
                reverse = True
            ratio = None
            for a, b in zip(base, strip):
                if not a and (not b):
                    continue
                elif not a or not b:
                    break
                elif b % a != 0:
                    break
                else:
                    _ratio = b // a
                    if ratio is None:
                        ratio = _ratio
                    elif ratio != _ratio:
                        break
            else:
                if reverse:
                    ratio = -ratio
                poly = poly.eval(gen, 1)
                coeff *= gen ** (-ratio)
                gens.remove(gen)
        if gens:
            poly = poly.eject(*gens)
    if poly.is_univariate and poly.get_domain().is_ZZ:
        basis = _integer_basis(poly)
        if basis is not None:
            n = poly.degree()

            def func(k, coeff):
                return coeff // basis ** (n - k[0])
            poly = poly.termwise(func)
            coeff *= basis
    if not isinstance(poly, poly_func):
        poly = poly_func(poly)
    return (coeff, poly)