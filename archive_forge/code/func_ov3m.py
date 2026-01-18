from functools import reduce
from itertools import combinations, chain
from math import factorial
from operator import mul
import numpy as np
from ase.units import kg, C, _hbar, kB
from ase.vibrations import Vibrations
def ov3m(self, m, delta):
    sum = delta * self.ov2m(m, delta) / np.sqrt(6.0)
    if m == 0:
        return sum
    else:
        assert m > 0
        return sum + np.sqrt(m / 3.0) * self.ov2m(m - 1, delta)