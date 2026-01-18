import numpy as np
from . import Filter  # prevent circular import in Python < 3.5
def papadakis(val, a):
    y = np.empty(np.shape(val))
    l1 = a
    l2 = a * 5.0 / 3
    r1ind = (val >= 0) * (val < l1)
    r2ind = (val >= l1) * (val < l2)
    r3ind = val >= l2
    y[r1ind] = 1
    y[r2ind] = np.sqrt((1 - np.sin(3 * np.pi / (2 * a) * val[r2ind])) / 2.0)
    y[r3ind] = 0
    return y