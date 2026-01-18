import math
from bisect import bisect
import sys
from .backend import (MPZ, MPZ_TYPE, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE,
from .libintmath import (giant_steps,
def python_mpf_mul(s, t, prec=0, rnd=round_fast):
    """Multiply two raw mpfs"""
    ssign, sman, sexp, sbc = s
    tsign, tman, texp, tbc = t
    sign = ssign ^ tsign
    man = sman * tman
    if man:
        bc = sbc + tbc - 1
        bc += int(man >> bc)
        if prec:
            return normalize1(sign, man, sexp + texp, bc, prec, rnd)
        else:
            return (sign, man, sexp + texp, bc)
    s_special = not sman and sexp
    t_special = not tman and texp
    if not s_special and (not t_special):
        return fzero
    if fnan in (s, t):
        return fnan
    if not tman and texp:
        s, t = (t, s)
    if t == fzero:
        return fnan
    return {1: finf, -1: fninf}[mpf_sign(s) * mpf_sign(t)]