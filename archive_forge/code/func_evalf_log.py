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
def evalf_log(expr: 'log', prec: int, options: OPT_DICT) -> TMP_RES:
    if len(expr.args) > 1:
        expr = expr.doit()
        return evalf(expr, prec, options)
    arg = expr.args[0]
    workprec = prec + 10
    result = evalf(arg, workprec, options)
    if result is S.ComplexInfinity:
        return result
    xre, xim, xacc, _ = result
    if xre is xim is None:
        xre = fzero
    if xim:
        from sympy.functions.elementary.complexes import Abs
        from sympy.functions.elementary.exponential import log
        re = evalf_log(log(Abs(arg, evaluate=False), evaluate=False), prec, options)
        im = mpf_atan2(xim, xre or fzero, prec)
        return (re[0], im, re[2], prec)
    imaginary_term = mpf_cmp(xre, fzero) < 0
    re = mpf_log(mpf_abs(xre), prec, rnd)
    size = fastlog(re)
    if prec - size > workprec and re != fzero:
        from .add import Add
        add = Add(S.NegativeOne, arg, evaluate=False)
        xre, xim, _, _ = evalf_add(add, prec, options)
        prec2 = workprec - fastlog(xre)
        re = mpf_log(mpf_abs(mpf_add(xre, fone, prec2)), prec, rnd)
    re_acc = prec
    if imaginary_term:
        return (re, mpf_pi(prec), re_acc, prec)
    else:
        return (re, None, re_acc, None)