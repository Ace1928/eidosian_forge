import math
from bisect import bisect
import sys
from .backend import (MPZ, MPZ_TYPE, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE,
from .libintmath import (giant_steps,
def mpf_div(s, t, prec, rnd=round_fast):
    """Floating-point division"""
    ssign, sman, sexp, sbc = s
    tsign, tman, texp, tbc = t
    if not sman or not tman:
        if s == fzero:
            if t == fzero:
                raise ZeroDivisionError
            if t == fnan:
                return fnan
            return fzero
        if t == fzero:
            raise ZeroDivisionError
        s_special = not sman and sexp
        t_special = not tman and texp
        if s_special and t_special:
            return fnan
        if s == fnan or t == fnan:
            return fnan
        if not t_special:
            if t == fzero:
                return fnan
            return {1: finf, -1: fninf}[mpf_sign(s) * mpf_sign(t)]
        return fzero
    sign = ssign ^ tsign
    if tman == 1:
        return normalize1(sign, sman, sexp - texp, sbc, prec, rnd)
    extra = prec - sbc + tbc + 5
    if extra < 5:
        extra = 5
    quot, rem = divmod(sman << extra, tman)
    if rem:
        quot = (quot << 1) + 1
        extra += 1
        return normalize1(sign, quot, sexp - texp - extra, bitcount(quot), prec, rnd)
    return normalize(sign, quot, sexp - texp - extra, bitcount(quot), prec, rnd)