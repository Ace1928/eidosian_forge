import math
from bisect import bisect
import sys
from .backend import (MPZ, MPZ_TYPE, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE,
from .libintmath import (giant_steps,
def mpf_cmp(s, t):
    """Compare the raw mpfs s and t. Return -1 if s < t, 0 if s == t,
    and 1 if s > t. (Same convention as Python's cmp() function.)"""
    ssign, sman, sexp, sbc = s
    tsign, tman, texp, tbc = t
    if not sman or not tman:
        if s == fzero:
            return -mpf_sign(t)
        if t == fzero:
            return mpf_sign(s)
        if s == t:
            return 0
        if t == fnan:
            return 1
        if s == finf:
            return 1
        if t == fninf:
            return 1
        return -1
    if ssign != tsign:
        if not ssign:
            return 1
        return -1
    if sexp == texp:
        if sman == tman:
            return 0
        if sman > tman:
            if ssign:
                return -1
            else:
                return 1
        elif ssign:
            return 1
        else:
            return -1
    a = sbc + sexp
    b = tbc + texp
    if ssign:
        if a < b:
            return 1
        if a > b:
            return -1
    else:
        if a < b:
            return -1
        if a > b:
            return 1
    delta = mpf_sub(s, t, 5, round_floor)
    if delta[0]:
        return -1
    return 1