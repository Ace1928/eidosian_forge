import math
from bisect import bisect
import sys
from .backend import (MPZ, MPZ_TYPE, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE,
from .libintmath import (giant_steps,
def mpf_add(s, t, prec=0, rnd=round_fast, _sub=0):
    """
    Add the two raw mpf values s and t.

    With prec=0, no rounding is performed. Note that this can
    produce a very large mantissa (potentially too large to fit
    in memory) if exponents are far apart.
    """
    ssign, sman, sexp, sbc = s
    tsign, tman, texp, tbc = t
    tsign ^= _sub
    if sman and tman:
        offset = sexp - texp
        if offset:
            if offset > 0:
                if offset > 100 and prec:
                    delta = sbc + sexp - tbc - texp
                    if delta > prec + 4:
                        offset = prec + 4
                        sman <<= offset
                        if tsign == ssign:
                            sman += 1
                        else:
                            sman -= 1
                        return normalize1(ssign, sman, sexp - offset, bitcount(sman), prec, rnd)
                if ssign == tsign:
                    man = tman + (sman << offset)
                else:
                    if ssign:
                        man = tman - (sman << offset)
                    else:
                        man = (sman << offset) - tman
                    if man >= 0:
                        ssign = 0
                    else:
                        man = -man
                        ssign = 1
                bc = bitcount(man)
                return normalize1(ssign, man, texp, bc, prec or bc, rnd)
            elif offset < 0:
                if offset < -100 and prec:
                    delta = tbc + texp - sbc - sexp
                    if delta > prec + 4:
                        offset = prec + 4
                        tman <<= offset
                        if ssign == tsign:
                            tman += 1
                        else:
                            tman -= 1
                        return normalize1(tsign, tman, texp - offset, bitcount(tman), prec, rnd)
                if ssign == tsign:
                    man = sman + (tman << -offset)
                else:
                    if tsign:
                        man = sman - (tman << -offset)
                    else:
                        man = (tman << -offset) - sman
                    if man >= 0:
                        ssign = 0
                    else:
                        man = -man
                        ssign = 1
                bc = bitcount(man)
                return normalize1(ssign, man, sexp, bc, prec or bc, rnd)
        if ssign == tsign:
            man = tman + sman
        else:
            if ssign:
                man = tman - sman
            else:
                man = sman - tman
            if man >= 0:
                ssign = 0
            else:
                man = -man
                ssign = 1
        bc = bitcount(man)
        return normalize(ssign, man, texp, bc, prec or bc, rnd)
    if _sub:
        t = mpf_neg(t)
    if not sman:
        if sexp:
            if s == t or tman or (not texp):
                return s
            return fnan
        if tman:
            return normalize1(tsign, tman, texp, tbc, prec or tbc, rnd)
        return t
    if texp:
        return t
    if sman:
        return normalize1(ssign, sman, sexp, sbc, prec or sbc, rnd)
    return s