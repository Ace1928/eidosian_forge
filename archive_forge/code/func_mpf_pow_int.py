import math
from bisect import bisect
import sys
from .backend import (MPZ, MPZ_TYPE, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE,
from .libintmath import (giant_steps,
def mpf_pow_int(s, n, prec, rnd=round_fast):
    """Compute s**n, where s is a raw mpf and n is a Python integer."""
    sign, man, exp, bc = s
    if not man and exp:
        if s == finf:
            if n > 0:
                return s
            if n == 0:
                return fnan
            return fzero
        if s == fninf:
            if n > 0:
                return [finf, fninf][n & 1]
            if n == 0:
                return fnan
            return fzero
        return fnan
    n = int(n)
    if n == 0:
        return fone
    if n == 1:
        return mpf_pos(s, prec, rnd)
    if n == 2:
        _, man, exp, bc = s
        if not man:
            return fzero
        man = man * man
        if man == 1:
            return (0, MPZ_ONE, exp + exp, 1)
        bc = bc + bc - 2
        bc += bctable[int(man >> bc)]
        return normalize1(0, man, exp + exp, bc, prec, rnd)
    if n == -1:
        return mpf_div(fone, s, prec, rnd)
    if n < 0:
        inverse = mpf_pow_int(s, -n, prec + 5, reciprocal_rnd[rnd])
        return mpf_div(fone, inverse, prec, rnd)
    result_sign = sign & n
    if man == 1:
        return (result_sign, MPZ_ONE, exp * n, 1)
    if bc * n < 1000:
        man **= n
        return normalize1(result_sign, man, exp * n, bitcount(man), prec, rnd)
    rounds_down = rnd == round_nearest or shifts_down[rnd][result_sign]
    workprec = prec + 4 * bitcount(n) + 4
    _, pm, pe, pbc = fone
    while 1:
        if n & 1:
            pm = pm * man
            pe = pe + exp
            pbc += bc - 2
            pbc = pbc + bctable[int(pm >> pbc)]
            if pbc > workprec:
                if rounds_down:
                    pm = pm >> pbc - workprec
                else:
                    pm = -(-pm >> pbc - workprec)
                pe += pbc - workprec
                pbc = workprec
            n -= 1
            if not n:
                break
        man = man * man
        exp = exp + exp
        bc = bc + bc - 2
        bc = bc + bctable[int(man >> bc)]
        if bc > workprec:
            if rounds_down:
                man = man >> bc - workprec
            else:
                man = -(-man >> bc - workprec)
            exp += bc - workprec
            bc = workprec
        n = n // 2
    return normalize(result_sign, pm, pe, pbc, prec, rnd)