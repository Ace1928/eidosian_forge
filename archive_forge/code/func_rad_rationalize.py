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
def rad_rationalize(num, den):
    """
    Rationalize ``num/den`` by removing square roots in the denominator;
    num and den are sum of terms whose squares are positive rationals.

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.simplify.radsimp import rad_rationalize
    >>> rad_rationalize(sqrt(3), 1 + sqrt(2)/3)
    (-sqrt(3) + sqrt(6)/3, -7/9)
    """
    if not den.is_Add:
        return (num, den)
    g, a, b = split_surds(den)
    a = a * sqrt(g)
    num = _mexpand((a - b) * num)
    den = _mexpand(a ** 2 - b ** 2)
    return rad_rationalize(num, den)