import functools
from threading import RLock
import numpy as np
from scipy.optimize import _cobyla as cobyla
from ._optimize import (OptimizeResult, _check_unknown_options,
def calcfc(x, con):
    f = sf.fun(x)
    i = 0
    for size, c in izip(cons_lengths, constraints):
        con[i:i + size] = c['fun'](x, *c['args'])
        i += size
    return f