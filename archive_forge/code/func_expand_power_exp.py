from __future__ import annotations
from typing import Any
from collections.abc import Iterable
from .add import Add
from .basic import Basic, _atomic
from .cache import cacheit
from .containers import Tuple, Dict
from .decorators import _sympifyit
from .evalf import pure_complex
from .expr import Expr, AtomicExpr
from .logic import fuzzy_and, fuzzy_or, fuzzy_not, FuzzyBool
from .mul import Mul
from .numbers import Rational, Float, Integer
from .operations import LatticeOp
from .parameters import global_parameters
from .rules import Transform
from .singleton import S
from .sympify import sympify, _sympify
from .sorting import default_sort_key, ordered
from sympy.utilities.exceptions import (sympy_deprecation_warning,
from sympy.utilities.iterables import (has_dups, sift, iterable,
from sympy.utilities.lambdify import MPMATH_TRANSLATIONS
from sympy.utilities.misc import as_int, filldedent, func_name
import mpmath
from mpmath.libmp.libmpf import prec_to_dps
import inspect
from collections import Counter
from .symbol import Dummy, Symbol
def expand_power_exp(expr, deep=True):
    """
    Wrapper around expand that only uses the power_exp hint.

    See the expand docstring for more information.

    Examples
    ========

    >>> from sympy import expand_power_exp, Symbol
    >>> from sympy.abc import x, y
    >>> expand_power_exp(3**(y + 2))
    9*3**y
    >>> expand_power_exp(x**(y + 2))
    x**(y + 2)

    If ``x = 0`` the value of the expression depends on the
    value of ``y``; if the expression were expanded the result
    would be 0. So expansion is only done if ``x != 0``:

    >>> expand_power_exp(Symbol('x', zero=False)**(y + 2))
    x**2*x**y
    """
    return sympify(expr).expand(deep=deep, complex=False, basic=False, log=False, mul=False, power_exp=True, power_base=False, multinomial=False)