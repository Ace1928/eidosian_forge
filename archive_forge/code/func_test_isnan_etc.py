import mpmath
from mpmath import *
from mpmath.libmp import *
import random
import sys
def test_isnan_etc():
    from mpmath.rational import mpq
    assert isnan(nan) == True
    assert isnan(3) == False
    assert isnan(mpf(3)) == False
    assert isnan(inf) == False
    assert isnan(mpc(2, nan)) == True
    assert isnan(mpc(2, nan)) == True
    assert isnan(mpc(nan, nan)) == True
    assert isnan(mpc(2, 2)) == False
    assert isnan(mpc(nan, inf)) == True
    assert isnan(mpc(inf, inf)) == False
    assert isnan(mpq((3, 2))) == False
    assert isnan(mpq((0, 1))) == False
    assert isinf(inf) == True
    assert isinf(-inf) == True
    assert isinf(3) == False
    assert isinf(nan) == False
    assert isinf(3 + 4j) == False
    assert isinf(mpc(inf)) == True
    assert isinf(mpc(3, inf)) == True
    assert isinf(mpc(inf, 3)) == True
    assert isinf(mpc(inf, inf)) == True
    assert isinf(mpc(nan, inf)) == True
    assert isinf(mpc(inf, nan)) == True
    assert isinf(mpc(nan, nan)) == False
    assert isinf(mpq((3, 2))) == False
    assert isinf(mpq((0, 1))) == False
    assert isnormal(3) == True
    assert isnormal(3.5) == True
    assert isnormal(mpf(3.5)) == True
    assert isnormal(0) == False
    assert isnormal(mpf(0)) == False
    assert isnormal(0.0) == False
    assert isnormal(inf) == False
    assert isnormal(-inf) == False
    assert isnormal(nan) == False
    assert isnormal(float(inf)) == False
    assert isnormal(mpc(0, 0)) == False
    assert isnormal(mpc(3, 0)) == True
    assert isnormal(mpc(0, 3)) == True
    assert isnormal(mpc(3, 3)) == True
    assert isnormal(mpc(0, nan)) == False
    assert isnormal(mpc(0, inf)) == False
    assert isnormal(mpc(3, nan)) == False
    assert isnormal(mpc(3, inf)) == False
    assert isnormal(mpc(3, -inf)) == False
    assert isnormal(mpc(nan, 0)) == False
    assert isnormal(mpc(inf, 0)) == False
    assert isnormal(mpc(nan, 3)) == False
    assert isnormal(mpc(inf, 3)) == False
    assert isnormal(mpc(inf, nan)) == False
    assert isnormal(mpc(nan, inf)) == False
    assert isnormal(mpc(nan, nan)) == False
    assert isnormal(mpc(inf, inf)) == False
    assert isnormal(mpq((3, 2))) == True
    assert isnormal(mpq((0, 1))) == False
    assert isint(3) == True
    assert isint(0) == True
    assert isint(long(3)) == True
    assert isint(long(0)) == True
    assert isint(mpf(3)) == True
    assert isint(mpf(0)) == True
    assert isint(mpf(-3)) == True
    assert isint(mpf(3.2)) == False
    assert isint(3.2) == False
    assert isint(nan) == False
    assert isint(inf) == False
    assert isint(-inf) == False
    assert isint(mpc(0)) == True
    assert isint(mpc(3)) == True
    assert isint(mpc(3.2)) == False
    assert isint(mpc(3, inf)) == False
    assert isint(mpc(inf)) == False
    assert isint(mpc(3, 2)) == False
    assert isint(mpc(0, 2)) == False
    assert isint(mpc(3, 2), gaussian=True) == True
    assert isint(mpc(3, 0), gaussian=True) == True
    assert isint(mpc(0, 3), gaussian=True) == True
    assert isint(3 + 4j) == False
    assert isint(3 + 4j, gaussian=True) == True
    assert isint(3 + 0j) == True
    assert isint(mpq((3, 2))) == False
    assert isint(mpq((3, 9))) == False
    assert isint(mpq((9, 3))) == True
    assert isint(mpq((0, 4))) == True
    assert isint(mpq((1, 1))) == True
    assert isint(mpq((-1, 1))) == True
    assert mp.isnpint(0) == True
    assert mp.isnpint(1) == False
    assert mp.isnpint(-1) == True
    assert mp.isnpint(-1.1) == False
    assert mp.isnpint(-1.0) == True
    assert mp.isnpint(mp.mpq(1, 2)) == False
    assert mp.isnpint(mp.mpq(-1, 2)) == False
    assert mp.isnpint(mp.mpq(-3, 1)) == True
    assert mp.isnpint(mp.mpq(0, 1)) == True
    assert mp.isnpint(mp.mpq(1, 1)) == False
    assert mp.isnpint(0 + 0j) == True
    assert mp.isnpint(-1 + 0j) == True
    assert mp.isnpint(-1.1 + 0j) == False
    assert mp.isnpint(-1 + 0.1j) == False
    assert mp.isnpint(0 + 0.1j) == False