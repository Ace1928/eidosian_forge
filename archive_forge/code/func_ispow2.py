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
def ispow2(d, log2=False):
    if not d.is_Pow:
        return False
    e = d.exp
    if e.is_Rational and e.q == 2 or (symbolic and denom(e) == 2):
        return True
    if log2:
        q = 1
        if e.is_Rational:
            q = e.q
        elif symbolic:
            d = denom(e)
            if d.is_Integer:
                q = d
        if q != 1 and log(q, 2).is_Integer:
            return True
    return False