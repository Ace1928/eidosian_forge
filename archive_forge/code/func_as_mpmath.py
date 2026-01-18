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
def as_mpmath(x: Any, prec: int, options: OPT_DICT) -> tUnion[mpc, mpf]:
    from .numbers import Infinity, NegativeInfinity, Zero
    x = sympify(x)
    if isinstance(x, Zero) or x == 0.0:
        return mpf(0)
    if isinstance(x, Infinity):
        return mpf('inf')
    if isinstance(x, NegativeInfinity):
        return mpf('-inf')
    result = evalf(x, prec, options)
    return quad_to_mpmath(result)