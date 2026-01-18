from functools import reduce
from itertools import combinations, chain
from math import factorial
from operator import mul
import numpy as np
from ase.units import kg, C, _hbar, kB
from ase.vibrations import Vibrations
def ov1mm2(self, m, delta):
    p1 = delta ** 3 / 4.0
    sum = p1 * self.ov0m(m, delta) ** 2
    if m == 0:
        return sum
    p2 = delta - 3.0 * delta ** 3 / 4
    sum += p2 * self.ov0m(m - 1, delta) ** 2
    if m == 1:
        return sum
    sum -= p2 * self.ov0m(m - 2, delta) ** 2
    if m == 2:
        return sum
    return sum - p1 * self.ov0m(m - 3, delta) ** 2