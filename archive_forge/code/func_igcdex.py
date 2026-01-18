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
def igcdex(a, b):
    """Returns x, y, g such that g = x*a + y*b = gcd(a, b).

    Examples
    ========

    >>> from sympy.core.numbers import igcdex
    >>> igcdex(2, 3)
    (-1, 1, 1)
    >>> igcdex(10, 12)
    (-1, 1, 2)

    >>> x, y, g = igcdex(100, 2004)
    >>> x, y, g
    (-20, 1, 4)
    >>> x*100 + y*2004
    4

    """
    if not a and (not b):
        return (0, 1, 0)
    if not a:
        return (0, b // abs(b), abs(b))
    if not b:
        return (a // abs(a), 0, abs(a))
    if a < 0:
        a, x_sign = (-a, -1)
    else:
        x_sign = 1
    if b < 0:
        b, y_sign = (-b, -1)
    else:
        y_sign = 1
    x, y, r, s = (1, 0, 0, 1)
    while b:
        c, q = (a % b, a // b)
        a, b, r, s, x, y = (b, c, x - q * r, y - q * s, r, s)
    return (x * x_sign, y * y_sign, a)