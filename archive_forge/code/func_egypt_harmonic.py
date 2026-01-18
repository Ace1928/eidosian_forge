from sympy.core.containers import Tuple
from sympy.core.numbers import (Integer, Rational)
from sympy.core.singleton import S
import sympy.polys
from math import gcd
def egypt_harmonic(r):
    rv = []
    d = S.One
    acc = S.Zero
    while acc + 1 / d <= r:
        acc += 1 / d
        rv.append(d)
        d += 1
    return (rv, r - acc)