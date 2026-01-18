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
def evalf_trig(v: 'Expr', prec: int, options: OPT_DICT) -> TMP_RES:
    """
    This function handles sin and cos of complex arguments.

    TODO: should also handle tan of complex arguments.
    """
    from sympy.functions.elementary.trigonometric import cos, sin
    if isinstance(v, cos):
        func = mpf_cos
    elif isinstance(v, sin):
        func = mpf_sin
    else:
        raise NotImplementedError
    arg = v.args[0]
    xprec = prec + 20
    re, im, re_acc, im_acc = evalf(arg, xprec, options)
    if im:
        if 'subs' in options:
            v = v.subs(options['subs'])
        return evalf(v._eval_evalf(prec), prec, options)
    if not re:
        if isinstance(v, cos):
            return (fone, None, prec, None)
        elif isinstance(v, sin):
            return (None, None, None, None)
        else:
            raise NotImplementedError
    xsize = fastlog(re)
    if xsize < 1:
        return (func(re, prec, rnd), None, prec, None)
    if xsize >= 10:
        xprec = prec + xsize
        re, im, re_acc, im_acc = evalf(arg, xprec, options)
    while 1:
        y = func(re, prec, rnd)
        ysize = fastlog(y)
        gap = -ysize
        accuracy = xprec - xsize - gap
        if accuracy < prec:
            if options.get('verbose'):
                print('SIN/COS', accuracy, 'wanted', prec, 'gap', gap)
                print(to_str(y, 10))
            if xprec > options.get('maxprec', DEFAULT_MAXPREC):
                return (y, None, accuracy, None)
            xprec += gap
            re, im, re_acc, im_acc = evalf(arg, xprec, options)
            continue
        else:
            return (y, None, prec, None)