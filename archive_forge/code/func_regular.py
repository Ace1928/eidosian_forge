import numpy as np
from . import Filter  # prevent circular import in Python < 3.5
def regular(val, d):
    if d == 0:
        return np.sin(np.pi / 4.0 * val)
    else:
        output = np.sin(np.pi * (val - 1) / 2.0)
        for i in range(2, d):
            output = np.sin(np.pi * output / 2.0)
        return np.sin(np.pi / 4.0 * (1 + output))