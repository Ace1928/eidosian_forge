import math
from bisect import bisect
import sys
from .backend import (MPZ, MPZ_TYPE, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE,
from .libintmath import (giant_steps,
def strict_normalize1(sign, man, exp, bc, prec, rnd):
    """Additional checks on the components of an mpf. Enable tests by setting
       the environment variable MPMATH_STRICT to Y."""
    assert type(man) == MPZ_TYPE
    assert type(bc) in _exp_types
    assert type(exp) in _exp_types
    assert bc == bitcount(man)
    assert not man or man & 1
    return _normalize1(sign, man, exp, bc, prec, rnd)