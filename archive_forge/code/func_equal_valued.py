from __future__ import annotations
import numbers
import decimal
import fractions
import math
import re as regex
import sys
from functools import lru_cache
from .containers import Tuple
from .sympify import (SympifyError, _sympy_converter, sympify, _convert_numpy_types,
from .singleton import S, Singleton
from .basic import Basic
from .expr import Expr, AtomicExpr
from .evalf import pure_complex
from .cache import cacheit, clear_cache
from .decorators import _sympifyit
from .logic import fuzzy_not
from .kind import NumberKind
from sympy.external.gmpy import SYMPY_INTS, HAS_GMPY, gmpy
from sympy.multipledispatch import dispatch
import mpmath
import mpmath.libmp as mlib
from mpmath.libmp import bitcount, round_nearest as rnd
from mpmath.libmp.backend import MPZ
from mpmath.libmp import mpf_pow, mpf_pi, mpf_e, phi_fixed
from mpmath.ctx_mp import mpnumeric
from mpmath.libmp.libmpf import (
from sympy.utilities.misc import as_int, debug, filldedent
from .parameters import global_parameters
from .power import Pow, integer_nthroot
from .mul import Mul
from .add import Add
def equal_valued(x, y):
    """Compare expressions treating plain floats as rationals.

    Examples
    ========

    >>> from sympy import S, symbols, Rational, Float
    >>> from sympy.core.numbers import equal_valued
    >>> equal_valued(1, 2)
    False
    >>> equal_valued(1, 1)
    True

    In SymPy expressions with Floats compare unequal to corresponding
    expressions with rationals:

    >>> x = symbols('x')
    >>> x**2 == x**2.0
    False

    However an individual Float compares equal to a Rational:

    >>> Rational(1, 2) == Float(0.5)
    True

    In a future version of SymPy this might change so that Rational and Float
    compare unequal. This function provides the behavior currently expected of
    ``==`` so that it could still be used if the behavior of ``==`` were to
    change in future.

    >>> equal_valued(1, 1.0) # Float vs Rational
    True
    >>> equal_valued(S(1).n(3), S(1).n(5)) # Floats of different precision
    True

    Explanation
    ===========

    In future SymPy verions Float and Rational might compare unequal and floats
    with different precisions might compare unequal. In that context a function
    is needed that can check if a number is equal to 1 or 0 etc. The idea is
    that instead of testing ``if x == 1:`` if we want to accept floats like
    ``1.0`` as well then the test can be written as ``if equal_valued(x, 1):``
    or ``if equal_valued(x, 2):``. Since this function is intended to be used
    in situations where one or both operands are expected to be concrete
    numbers like 1 or 0 the function does not recurse through the args of any
    compound expression to compare any nested floats.

    References
    ==========

    .. [1] https://github.com/sympy/sympy/pull/20033
    """
    x = _sympify(x)
    y = _sympify(y)
    if not x.is_Float and (not y.is_Float):
        return x == y
    elif x.is_Float and y.is_Float:
        return x._mpf_ == y._mpf_
    elif x.is_Float:
        x, y = (y, x)
    if not x.is_Rational:
        return False
    sign, man, exp, _ = y._mpf_
    p, q = (x.p, x.q)
    if sign:
        man = -man
    if exp == 0:
        return q == 1 and man == p
    elif exp > 0:
        if q != 1:
            return False
        if p.bit_length() != man.bit_length() + exp:
            return False
        return man << exp == p
    else:
        if p != man:
            return False
        neg_exp = -exp
        if q.bit_length() - 1 != neg_exp:
            return False
        return 1 << neg_exp == q