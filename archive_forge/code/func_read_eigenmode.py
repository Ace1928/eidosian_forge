import sys
import time
import warnings
from math import cos, sin, atan, tan, degrees, pi, sqrt
from typing import Dict, Any
import numpy as np
from ase.optimize.optimize import Optimizer
from ase.parallel import world
from ase.calculators.singlepoint import SinglePointCalculator
from ase.utils import IOContext
def read_eigenmode(mlog, index=-1):
    """Read an eigenmode.
    To access the pre optimization eigenmode set index = 'null'.

    """
    mlog_is_str = isinstance(mlog, str)
    if mlog_is_str:
        fd = open(mlog, 'r')
    else:
        fd = mlog
    lines = fd.readlines()
    k = 2
    while lines[k].split()[1].lower() not in ['optimization', 'order']:
        k += 1
    n = k - 2
    n_itr = len(lines) // (n + 1) - 2
    if isinstance(index, str):
        if index.lower() == 'null':
            i = 0
        else:
            i = int(index) + 1
    elif index >= 0:
        i = index + 1
    elif index < -n_itr - 1:
        raise IndexError('list index out of range')
    else:
        i = index
    mode = np.ndarray(shape=(n, 3), dtype=float)
    k_atom = 0
    for k in range(1, n + 1):
        line = lines[i * (n + 1) + k].split()
        for k_dim in range(3):
            mode[k_atom][k_dim] = float(line[k_dim + 2])
        k_atom += 1
    if mlog_is_str:
        fd.close()
    return mode