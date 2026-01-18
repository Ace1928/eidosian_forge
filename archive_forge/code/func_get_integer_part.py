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
def get_integer_part(expr: 'Expr', no: int, options: OPT_DICT, return_ints=False) -> tUnion[TMP_RES, tTuple[int, int]]:
    """
    With no = 1, computes ceiling(expr)
    With no = -1, computes floor(expr)

    Note: this function either gives the exact result or signals failure.
    """
    from sympy.functions.elementary.complexes import re, im
    assumed_size = 30
    result = evalf(expr, assumed_size, options)
    if result is S.ComplexInfinity:
        raise ValueError('Cannot get integer part of Complex Infinity')
    ire, iim, ire_acc, iim_acc = result
    if ire and iim:
        gap = max(fastlog(ire) - ire_acc, fastlog(iim) - iim_acc)
    elif ire:
        gap = fastlog(ire) - ire_acc
    elif iim:
        gap = fastlog(iim) - iim_acc
    elif return_ints:
        return (0, 0)
    else:
        return (None, None, None, None)
    margin = 10
    if gap >= -margin:
        prec = margin + assumed_size + gap
        ire, iim, ire_acc, iim_acc = evalf(expr, prec, options)
    else:
        prec = assumed_size

    def calc_part(re_im: 'Expr', nexpr: MPF_TUP):
        from .add import Add
        _, _, exponent, _ = nexpr
        is_int = exponent == 0
        nint = int(to_int(nexpr, rnd))
        if is_int:
            ire, iim, ire_acc, iim_acc = evalf(re_im - nint, 10, options)
            assert not iim
            size = -fastlog(ire) + 2
            if size > prec:
                ire, iim, ire_acc, iim_acc = evalf(re_im, size, options)
                assert not iim
                nexpr = ire
            nint = int(to_int(nexpr, rnd))
            _, _, new_exp, _ = ire
            is_int = new_exp == 0
        if not is_int:
            s = options.get('subs', False)
            if s:
                doit = True
                for v in s.values():
                    try:
                        as_int(v, strict=False)
                    except ValueError:
                        try:
                            [as_int(i, strict=False) for i in v.as_real_imag()]
                            continue
                        except (ValueError, AttributeError):
                            doit = False
                            break
                if doit:
                    re_im = re_im.subs(s)
            re_im = Add(re_im, -nint, evaluate=False)
            x, _, x_acc, _ = evalf(re_im, 10, options)
            try:
                check_target(re_im, (x, None, x_acc, None), 3)
            except PrecisionExhausted:
                if not re_im.equals(0):
                    raise PrecisionExhausted
                x = fzero
            nint += int(no * (mpf_cmp(x or fzero, fzero) == no))
        nint = from_int(nint)
        return (nint, INF)
    re_, im_, re_acc, im_acc = (None, None, None, None)
    if ire:
        re_, re_acc = calc_part(re(expr, evaluate=False), ire)
    if iim:
        im_, im_acc = calc_part(im(expr, evaluate=False), iim)
    if return_ints:
        return (int(to_int(re_ or fzero)), int(to_int(im_ or fzero)))
    return (re_, im_, re_acc, im_acc)