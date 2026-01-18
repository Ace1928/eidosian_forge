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
def evalf_mul(v: 'Mul', prec: int, options: OPT_DICT) -> TMP_RES:
    res = pure_complex(v)
    if res:
        _, h = res
        im, _, im_acc, _ = evalf(h, prec, options)
        return (None, im, None, im_acc)
    args = list(v.args)
    has_zero = False
    special = []
    from .numbers import Float
    for arg in args:
        result = evalf(arg, prec, options)
        if result is S.ComplexInfinity:
            special.append(result)
            continue
        if result[0] is None:
            if result[1] is None:
                has_zero = True
            continue
        num = Float._new(result[0], 1)
        if num is S.NaN:
            return (fnan, None, prec, None)
        if num.is_infinite:
            special.append(num)
    if special:
        if has_zero:
            return (fnan, None, prec, None)
        from .mul import Mul
        return evalf(Mul(*special), prec + 4, {})
    if has_zero:
        return (None, None, None, None)
    acc = prec
    working_prec = prec + len(args) + 5
    start = man, exp, bc = (MPZ(1), 0, 1)
    last = len(args)
    direction = 0
    args.append(S.One)
    complex_factors = []
    for i, arg in enumerate(args):
        if i != last and pure_complex(arg):
            args[-1] = (args[-1] * arg).expand()
            continue
        elif i == last and arg is S.One:
            continue
        re, im, re_acc, im_acc = evalf(arg, working_prec, options)
        if re and im:
            complex_factors.append((re, im, re_acc, im_acc))
            continue
        elif re:
            (s, m, e, b), w_acc = (re, re_acc)
        elif im:
            (s, m, e, b), w_acc = (im, im_acc)
            direction += 1
        else:
            return (None, None, None, None)
        direction += 2 * s
        man *= m
        exp += e
        bc += b
        while bc > 3 * working_prec:
            man >>= working_prec
            exp += working_prec
            bc -= working_prec
        acc = min(acc, w_acc)
    sign = (direction & 2) >> 1
    if not complex_factors:
        v = normalize(sign, man, exp, bitcount(man), prec, rnd)
        if direction & 1:
            return (None, v, None, acc)
        else:
            return (v, None, acc, None)
    else:
        if (man, exp, bc) != start:
            re, im = ((sign, man, exp, bitcount(man)), (0, MPZ(0), 0, 0))
            i0 = 0
        else:
            wre, wim, wre_acc, wim_acc = complex_factors[0]
            acc = min(acc, complex_accuracy((wre, wim, wre_acc, wim_acc)))
            re = wre
            im = wim
            i0 = 1
        for wre, wim, wre_acc, wim_acc in complex_factors[i0:]:
            acc = min(acc, complex_accuracy((wre, wim, wre_acc, wim_acc)))
            use_prec = working_prec
            A = mpf_mul(re, wre, use_prec)
            B = mpf_mul(mpf_neg(im), wim, use_prec)
            C = mpf_mul(re, wim, use_prec)
            D = mpf_mul(im, wre, use_prec)
            re = mpf_add(A, B, use_prec)
            im = mpf_add(C, D, use_prec)
        if options.get('verbose'):
            print('MUL: wanted', prec, 'accurate bits, got', acc)
        if direction & 1:
            re, im = (mpf_neg(im), re)
        return (re, im, acc, acc)