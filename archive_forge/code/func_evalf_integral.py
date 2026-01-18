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
def evalf_integral(expr: 'Integral', prec: int, options: OPT_DICT) -> TMP_RES:
    limits = expr.limits
    if len(limits) != 1 or len(limits[0]) != 3:
        raise NotImplementedError
    workprec = prec
    i = 0
    maxprec = options.get('maxprec', INF)
    while 1:
        result = do_integral(expr, workprec, options)
        accuracy = complex_accuracy(result)
        if accuracy >= prec:
            break
        if workprec >= maxprec:
            break
        if accuracy == -1:
            workprec *= 2
        else:
            workprec += max(prec, 2 ** i)
        workprec = min(workprec, maxprec)
        i += 1
    return result