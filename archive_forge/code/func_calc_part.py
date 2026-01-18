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