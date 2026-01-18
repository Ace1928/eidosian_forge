from sympy.core import (S, Add, Mul, Pow, Eq, Expr,
from sympy.core.exprtools import decompose_power, decompose_power_rat
from sympy.core.numbers import _illegal
from sympy.polys.polyerrors import PolynomialError, GeneratorsError
from sympy.polys.polyoptions import build_options
import re
def expr_from_dict(rep, *gens):
    """Convert a multinomial form into an expression. """
    result = []
    for monom, coeff in rep.items():
        term = [coeff]
        for g, m in zip(gens, monom):
            if m:
                term.append(Pow(g, m))
        result.append(Mul(*term))
    return Add(*result)