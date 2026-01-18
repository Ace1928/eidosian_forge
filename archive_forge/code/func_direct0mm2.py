from functools import reduce
from itertools import combinations, chain
from math import factorial
from operator import mul
import numpy as np
from ase.units import kg, C, _hbar, kB
from ase.vibrations import Vibrations
def direct0mm2(self, m, delta):
    """direct and fast <0|m><m|2>"""
    S = delta ** 2 / 2.0
    sum = S ** (m + 1)
    if m >= 1:
        sum -= 2 * m * S ** m
    if m >= 2:
        sum += m * (m - 1) * S ** (m - 1)
    return np.exp(-S) / np.sqrt(2) * sum * self.factorial.inv(m)