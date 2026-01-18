from sympy.core.containers import Tuple
from sympy.core.numbers import (Integer, Rational)
from sympy.core.singleton import S
import sympy.polys
from math import gcd
def egypt_graham_jewett(x, y):
    l = [y] * x
    while len(l) != len(set(l)):
        l.sort()
        for i in range(len(l) - 1):
            if l[i] == l[i + 1]:
                break
        l[i + 1] = l[i] + 1
        l.append(l[i] * (l[i] + 1))
    return sorted(l)