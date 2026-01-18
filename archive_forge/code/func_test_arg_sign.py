from mpmath.libmp import *
from mpmath import *
import random
import time
import math
import cmath
def test_arg_sign():
    assert arg(3) == 0
    assert arg(-3).ae(pi)
    assert arg(j).ae(pi / 2)
    assert arg(-j).ae(-pi / 2)
    assert arg(0) == 0
    assert isnan(atan2(3, nan))
    assert isnan(atan2(nan, 3))
    assert isnan(atan2(0, nan))
    assert isnan(atan2(nan, 0))
    assert isnan(atan2(nan, nan))
    assert arg(inf) == 0
    assert arg(-inf).ae(pi)
    assert isnan(arg(nan))
    assert sign(0) == 0
    assert sign(3) == 1
    assert sign(-3) == -1
    assert sign(inf) == 1
    assert sign(-inf) == -1
    assert isnan(sign(nan))
    assert sign(j) == j
    assert sign(-3 * j) == -j
    assert sign(1 + j).ae((1 + j) / sqrt(2))