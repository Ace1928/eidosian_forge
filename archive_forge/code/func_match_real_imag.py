from itertools import product
from typing import Tuple as tTuple
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.expr import Expr
from sympy.core.function import (Function, ArgumentIndexError, expand_log,
from sympy.core.logic import fuzzy_and, fuzzy_not, fuzzy_or
from sympy.core.mul import Mul
from sympy.core.numbers import Integer, Rational, pi, I, ImaginaryUnit
from sympy.core.parameters import global_parameters
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Wild, Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import arg, unpolarify, im, re, Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.ntheory import multiplicity, perfect_power
from sympy.ntheory.factor_ import factorint
def match_real_imag(expr):
    """
    Try to match expr with $a + Ib$ for real $a$ and $b$.

    ``match_real_imag`` returns a tuple containing the real and imaginary
    parts of expr or ``(None, None)`` if direct matching is not possible. Contrary
    to :func:`~.re()`, :func:`~.im()``, and ``as_real_imag()``, this helper will not force things
    by returning expressions themselves containing ``re()`` or ``im()`` and it
    does not expand its argument either.

    """
    r_, i_ = expr.as_independent(I, as_Add=True)
    if i_ == 0 and r_.is_real:
        return (r_, i_)
    i_ = i_.as_coefficient(I)
    if i_ and i_.is_real and r_.is_real:
        return (r_, i_)
    else:
        return (None, None)