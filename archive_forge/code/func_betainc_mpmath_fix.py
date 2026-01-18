from sympy.core import S
from sympy.core.function import Function, ArgumentIndexError
from sympy.core.symbol import Dummy
from sympy.functions.special.gamma_functions import gamma, digamma
from sympy.functions.combinatorial.numbers import catalan
from sympy.functions.elementary.complexes import conjugate
def betainc_mpmath_fix(a, b, x1, x2, reg=0):
    from mpmath import betainc, mpf
    if x1 == x2:
        return mpf(0)
    else:
        return betainc(a, b, x1, x2, reg)