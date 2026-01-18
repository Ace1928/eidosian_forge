import math
import sys
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_THREE, gmpy
from .libintmath import list_primes, ifac, ifac2, moebius
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
def complex_stirling_series(x, y, prec):
    _m = x * x + y * y >> prec
    tre = (x << prec) // _m
    tim = (-y << prec) // _m
    ure = tre * tre - tim * tim >> prec
    uim = tim * tre >> prec - 1
    sre = ln_sqrt2pi_fixed(prec) - x
    sim = -y
    sre += tre // 12
    sim += tim // 12
    tre, tim = (tre * ure - tim * uim >> prec, tre * uim + tim * ure >> prec)
    sre -= tre // 360
    sim -= tim // 360
    tre, tim = (tre * ure - tim * uim >> prec, tre * uim + tim * ure >> prec)
    sre += tre // 1260
    sim += tim // 1260
    tre, tim = (tre * ure - tim * uim >> prec, tre * uim + tim * ure >> prec)
    sre -= tre // 1680
    sim -= tim // 1680
    tre, tim = (tre * ure - tim * uim >> prec, tre * uim + tim * ure >> prec)
    if abs(tre) + abs(tim) < 5:
        return (sre, sim)
    sre += tre // 1188
    sim += tim // 1188
    tre, tim = (tre * ure - tim * uim >> prec, tre * uim + tim * ure >> prec)
    sre -= 691 * tre // 360360
    sim -= 691 * tim // 360360
    tre, tim = (tre * ure - tim * uim >> prec, tre * uim + tim * ure >> prec)
    sre += tre // 156
    sim += tim // 156
    tre, tim = (tre * ure - tim * uim >> prec, tre * uim + tim * ure >> prec)
    if abs(tre) + abs(tim) < 5:
        return (sre, sim)
    sre -= 3617 * tre // 122400
    sim -= 3617 * tim // 122400
    tre, tim = (tre * ure - tim * uim >> prec, tre * uim + tim * ure >> prec)
    sre += 43867 * tre // 244188
    sim += 43867 * tim // 244188
    tre, tim = (tre * ure - tim * uim >> prec, tre * uim + tim * ure >> prec)
    sre -= 174611 * tre // 125400
    sim -= 174611 * tim // 125400
    tre, tim = (tre * ure - tim * uim >> prec, tre * uim + tim * ure >> prec)
    if abs(tre) + abs(tim) < 5:
        return (sre, sim)
    k = 22
    usize = bitcount(max(abs(ure), abs(uim)))
    tsize = bitcount(max(abs(tre), abs(tim)))
    texp = 0
    while 1:
        p, q, pb, qb = stirling_coefficient(k)
        term_mag = tsize + pb + texp
        shift = -texp
        m = pb - term_mag
        if m > 0 and shift < m:
            p >>= m
            shift -= m
        m = tsize - term_mag
        if m > 0 and shift < m:
            wre = tre >> m
            wim = tim >> m
            shift -= m
        else:
            wre = tre
            wim = tim
        termre = tre * p // q >> shift
        termim = tim * p // q >> shift
        if abs(termre) + abs(termim) < 5:
            break
        sre += termre
        sim += termim
        tre, tim = (tre * ure - tim * uim >> usize, tre * uim + tim * ure >> usize)
        texp -= prec - usize
        k += 2
    return (sre, sim)