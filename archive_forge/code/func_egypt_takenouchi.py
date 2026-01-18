from sympy.core.containers import Tuple
from sympy.core.numbers import (Integer, Rational)
from sympy.core.singleton import S
import sympy.polys
from math import gcd
def egypt_takenouchi(x, y):
    if x == 3:
        if y % 2 == 0:
            return [y // 2, y]
        i = (y - 1) // 2
        j = i + 1
        k = j + i
        return [j, k, j * k]
    l = [y] * x
    while len(l) != len(set(l)):
        l.sort()
        for i in range(len(l) - 1):
            if l[i] == l[i + 1]:
                break
        k = l[i]
        if k % 2 == 0:
            l[i] = l[i] // 2
            del l[i + 1]
        else:
            l[i], l[i + 1] = ((k + 1) // 2, k * (k + 1) // 2)
    return sorted(l)