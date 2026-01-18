from functools import reduce
from itertools import combinations, chain
from math import factorial
from operator import mul
import numpy as np
from ase.units import kg, C, _hbar, kB
from ase.vibrations import Vibrations
def directT0(self, n, S):
    """|<0|n>|^2

        Direct squared Franck-Condon overlap corresponding to T=0.
        """
    return np.exp(-S) * S ** n * self.factorial.inv(n)