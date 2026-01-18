from __future__ import annotations
from typing import Tuple as tTuple, Optional, Union as tUnion, Callable, List, Dict as tDict, Type, TYPE_CHECKING, \
import math
import mpmath.libmp as libmp
from mpmath import (
from mpmath import inf as mpmath_inf
from mpmath.libmp import (from_int, from_man_exp, from_rational, fhalf,
from mpmath.libmp import bitcount as mpmath_bitcount
from mpmath.libmp.backend import MPZ
from mpmath.libmp.libmpc import _infs_nan
from mpmath.libmp.libmpf import dps_to_prec, prec_to_dps
from .sympify import sympify
from .singleton import S
from sympy.external.gmpy import SYMPY_INTS
from sympy.utilities.iterables import is_sequence
from sympy.utilities.lambdify import lambdify
from sympy.utilities.misc import as_int
def check_convergence(numer: 'Expr', denom: 'Expr', n: 'Symbol') -> tTuple[int, Any, Any]:
    """
    Returns
    =======

    (h, g, p) where
    -- h is:
        > 0 for convergence of rate 1/factorial(n)**h
        < 0 for divergence of rate factorial(n)**(-h)
        = 0 for geometric or polynomial convergence or divergence

    -- abs(g) is:
        > 1 for geometric convergence of rate 1/h**n
        < 1 for geometric divergence of rate h**n
        = 1 for polynomial convergence or divergence

        (g < 0 indicates an alternating series)

    -- p is:
        > 1 for polynomial convergence of rate 1/n**h
        <= 1 for polynomial divergence of rate n**(-h)

    """
    from sympy.polys.polytools import Poly
    npol = Poly(numer, n)
    dpol = Poly(denom, n)
    p = npol.degree()
    q = dpol.degree()
    rate = q - p
    if rate:
        return (rate, None, None)
    constant = dpol.LC() / npol.LC()
    from .numbers import equal_valued
    if not equal_valued(abs(constant), 1):
        return (rate, constant, None)
    if npol.degree() == dpol.degree() == 0:
        return (rate, constant, 0)
    pc = npol.all_coeffs()[1]
    qc = dpol.all_coeffs()[1]
    return (rate, constant, (qc - pc) / dpol.LC())