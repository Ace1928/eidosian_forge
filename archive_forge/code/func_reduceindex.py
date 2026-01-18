import math
from typing import Optional, Sequence
import numpy as np
from ase.atoms import Atoms
import ase.data
def reduceindex(M):
    """Reduce Miller index to the lowest equivalent integers."""
    oldM = M
    g = math.gcd(M[0], M[1])
    h = math.gcd(g, M[2])
    while h != 1:
        if h == 0:
            raise ValueError('Division by zero: Are the miller indices linearly dependent?')
        M = M // h
        g = math.gcd(M[0], M[1])
        h = math.gcd(g, M[2])
    if np.dot(oldM, M) > 0:
        return M
    else:
        return -M