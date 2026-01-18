import math
import sys
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_THREE, gmpy
from .libintmath import list_primes, ifac, ifac2, moebius
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
def mpf_bernoulli(n, prec, rnd=None):
    """Computation of Bernoulli numbers (numerically)"""
    if n < 2:
        if n < 0:
            raise ValueError('Bernoulli numbers only defined for n >= 0')
        if n == 0:
            return fone
        if n == 1:
            return mpf_neg(fhalf)
    if n & 1:
        return fzero
    if prec > BERNOULLI_PREC_CUTOFF and prec > bernoulli_size(n) * 1.1 + 1000:
        p, q = bernfrac(n)
        return from_rational(p, q, prec, rnd or round_floor)
    if n > MAX_BERNOULLI_CACHE:
        return mpf_bernoulli_huge(n, prec, rnd)
    wp = prec + 30
    wp += 32 - (prec & 31)
    cached = bernoulli_cache.get(wp)
    if cached:
        numbers, state = cached
        if n in numbers:
            if not rnd:
                return numbers[n]
            return mpf_pos(numbers[n], prec, rnd)
        m, bin, bin1 = state
        if n - m > 10:
            return mpf_bernoulli_huge(n, prec, rnd)
    else:
        if n > 10:
            return mpf_bernoulli_huge(n, prec, rnd)
        numbers = {0: fone}
        m, bin, bin1 = state = [2, MPZ(10), MPZ_ONE]
        bernoulli_cache[wp] = (numbers, state)
    while m <= n:
        case = m % 6
        szbm = bernoulli_size(m)
        s = 0
        sexp = max(0, szbm) - wp
        if m < 6:
            a = MPZ_ZERO
        else:
            a = bin1
        for j in xrange(1, m // 6 + 1):
            usign, uman, uexp, ubc = u = numbers[m - 6 * j]
            if usign:
                uman = -uman
            s += lshift(a * uman, uexp - sexp)
            j6 = 6 * j
            a *= (m - 5 - j6) * (m - 4 - j6) * (m - 3 - j6) * (m - 2 - j6) * (m - 1 - j6) * (m - j6)
            a //= (4 + j6) * (5 + j6) * (6 + j6) * (7 + j6) * (8 + j6) * (9 + j6)
        if case == 0:
            b = mpf_rdiv_int(m + 3, f3, wp)
        if case == 2:
            b = mpf_rdiv_int(m + 3, f3, wp)
        if case == 4:
            b = mpf_rdiv_int(-m - 3, f6, wp)
        s = from_man_exp(s, sexp, wp)
        b = mpf_div(mpf_sub(b, s, wp), from_int(bin), wp)
        numbers[m] = b
        m += 2
        bin = bin * ((m + 2) * (m + 3)) // (m * (m - 1))
        if m > 6:
            bin1 = bin1 * ((2 + m) * (3 + m)) // ((m - 7) * (m - 6))
        state[:] = [m, bin, bin1]
    return numbers[n]