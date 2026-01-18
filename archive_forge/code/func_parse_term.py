from collections import defaultdict
from sympy.core import sympify, S, Mul, Derivative, Pow
from sympy.core.add import _unevaluated_Add, Add
from sympy.core.assumptions import assumptions
from sympy.core.exprtools import Factors, gcd_terms
from sympy.core.function import _mexpand, expand_mul, expand_power_base
from sympy.core.mul import _keep_coeff, _unevaluated_Mul, _mulsort
from sympy.core.numbers import Rational, zoo, nan
from sympy.core.parameters import global_parameters
from sympy.core.sorting import ordered, default_sort_key
from sympy.core.symbol import Dummy, Wild, symbols
from sympy.functions import exp, sqrt, log
from sympy.functions.elementary.complexes import Abs
from sympy.polys import gcd
from sympy.simplify.sqrtdenest import sqrtdenest
from sympy.utilities.iterables import iterable, sift
def parse_term(expr):
    """Parses expression expr and outputs tuple (sexpr, rat_expo,
        sym_expo, deriv)
        where:
         - sexpr is the base expression
         - rat_expo is the rational exponent that sexpr is raised to
         - sym_expo is the symbolic exponent that sexpr is raised to
         - deriv contains the derivatives of the expression

         For example, the output of x would be (x, 1, None, None)
         the output of 2**x would be (2, 1, x, None).
        """
    rat_expo, sym_expo = (S.One, None)
    sexpr, deriv = (expr, None)
    if expr.is_Pow:
        if isinstance(expr.base, Derivative):
            sexpr, deriv = parse_derivative(expr.base)
        else:
            sexpr = expr.base
        if expr.base == S.Exp1:
            arg = expr.exp
            if arg.is_Rational:
                sexpr, rat_expo = (S.Exp1, arg)
            elif arg.is_Mul:
                coeff, tail = arg.as_coeff_Mul(rational=True)
                sexpr, rat_expo = (exp(tail), coeff)
        elif expr.exp.is_Number:
            rat_expo = expr.exp
        else:
            coeff, tail = expr.exp.as_coeff_Mul()
            if coeff.is_Number:
                rat_expo, sym_expo = (coeff, tail)
            else:
                sym_expo = expr.exp
    elif isinstance(expr, exp):
        arg = expr.exp
        if arg.is_Rational:
            sexpr, rat_expo = (S.Exp1, arg)
        elif arg.is_Mul:
            coeff, tail = arg.as_coeff_Mul(rational=True)
            sexpr, rat_expo = (exp(tail), coeff)
    elif isinstance(expr, Derivative):
        sexpr, deriv = parse_derivative(expr)
    return (sexpr, rat_expo, sym_expo, deriv)