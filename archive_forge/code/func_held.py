import numpy as np
from . import Filter  # prevent circular import in Python < 3.5
def held(val, a):
    y = np.empty(np.shape(val))
    l1 = a
    l2 = 2 * a
    mu = lambda x: -1.0 + 24.0 * x - 144.0 * x ** 2 + 256 * x ** 3
    r1ind = (val >= 0) * (val < l1)
    r2ind = (val >= l1) * (val < l2)
    r3ind = val >= l2
    y[r1ind] = 1
    y[r2ind] = np.sin(2 * np.pi * mu(val[r2ind] / (8.0 * a)))
    y[r3ind] = 0
    return y