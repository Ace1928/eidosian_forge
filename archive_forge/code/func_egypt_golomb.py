from sympy.core.containers import Tuple
from sympy.core.numbers import (Integer, Rational)
from sympy.core.singleton import S
import sympy.polys
from math import gcd
def egypt_golomb(x, y):
    if x == 1:
        return [y]
    xp = sympy.polys.ZZ.invert(int(x), int(y))
    rv = [xp * y]
    rv.extend(egypt_golomb((x * xp - 1) // y, xp))
    return sorted(rv)