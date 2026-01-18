from functools import reduce
from itertools import combinations, chain
from math import factorial
from operator import mul
import numpy as np
from ase.units import kg, C, _hbar, kB
from ase.vibrations import Vibrations
def direct1mm2(self, m, delta):
    S = delta ** 2 / 2.0
    sum = S ** 2
    if m > 0:
        sum -= 2 * m * S
    if m > 1:
        sum += m * (m - 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(S == 0, 0, np.exp(-S) * S ** (m - 1) / delta * (S - m) * sum * self.factorial.inv(m))