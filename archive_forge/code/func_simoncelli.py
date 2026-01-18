import numpy as np
from . import Filter  # prevent circular import in Python < 3.5
def simoncelli(val, a):
    y = np.empty(np.shape(val))
    l1 = a
    l2 = 2 * a
    r1ind = (val >= 0) * (val < l1)
    r2ind = (val >= l1) * (val < l2)
    r3ind = val >= l2
    y[r1ind] = 1
    y[r2ind] = np.cos(np.pi / 2 * np.log(val[r2ind] / float(a)) / np.log(2))
    y[r3ind] = 0
    return y