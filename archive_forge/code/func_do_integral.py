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
def do_integral(expr: 'Integral', prec: int, options: OPT_DICT) -> TMP_RES:
    func = expr.args[0]
    x, xlow, xhigh = expr.args[1]
    if xlow == xhigh:
        xlow = xhigh = 0
    elif x not in func.free_symbols:
        if xhigh.free_symbols & xlow.free_symbols:
            diff = xhigh - xlow
            if diff.is_number:
                xlow, xhigh = (0, diff)
    oldmaxprec = options.get('maxprec', DEFAULT_MAXPREC)
    options['maxprec'] = min(oldmaxprec, 2 * prec)
    with workprec(prec + 5):
        xlow = as_mpmath(xlow, prec + 15, options)
        xhigh = as_mpmath(xhigh, prec + 15, options)
        from sympy.functions.elementary.trigonometric import cos, sin
        from .symbol import Wild
        have_part = [False, False]
        max_real_term: tUnion[float, int] = MINUS_INF
        max_imag_term: tUnion[float, int] = MINUS_INF

        def f(t: 'Expr') -> tUnion[mpc, mpf]:
            nonlocal max_real_term, max_imag_term
            re, im, re_acc, im_acc = evalf(func, mp.prec, {'subs': {x: t}})
            have_part[0] = re or have_part[0]
            have_part[1] = im or have_part[1]
            max_real_term = max(max_real_term, fastlog(re))
            max_imag_term = max(max_imag_term, fastlog(im))
            if im:
                return mpc(re or fzero, im)
            return mpf(re or fzero)
        if options.get('quad') == 'osc':
            A = Wild('A', exclude=[x])
            B = Wild('B', exclude=[x])
            D = Wild('D')
            m = func.match(cos(A * x + B) * D)
            if not m:
                m = func.match(sin(A * x + B) * D)
            if not m:
                raise ValueError('An integrand of the form sin(A*x+B)*f(x) or cos(A*x+B)*f(x) is required for oscillatory quadrature')
            period = as_mpmath(2 * S.Pi / m[A], prec + 15, options)
            result = quadosc(f, [xlow, xhigh], period=period)
            quadrature_error = MINUS_INF
        else:
            result, quadrature_err = quadts(f, [xlow, xhigh], error=1)
            quadrature_error = fastlog(quadrature_err._mpf_)
    options['maxprec'] = oldmaxprec
    if have_part[0]:
        re: Optional[MPF_TUP] = result.real._mpf_
        re_acc: Optional[int]
        if re == fzero:
            re_s, re_acc = scaled_zero(int(-max(prec, max_real_term, quadrature_error)))
            re = scaled_zero(re_s)
        else:
            re_acc = int(-max(max_real_term - fastlog(re) - prec, quadrature_error))
    else:
        re, re_acc = (None, None)
    if have_part[1]:
        im: Optional[MPF_TUP] = result.imag._mpf_
        im_acc: Optional[int]
        if im == fzero:
            im_s, im_acc = scaled_zero(int(-max(prec, max_imag_term, quadrature_error)))
            im = scaled_zero(im_s)
        else:
            im_acc = int(-max(max_imag_term - fastlog(im) - prec, quadrature_error))
    else:
        im, im_acc = (None, None)
    result = (re, im, re_acc, im_acc)
    return result