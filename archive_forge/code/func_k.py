import numpy as np
from . import Filter  # prevent circular import in Python < 3.5
def k(x):
    return np.sin(0.5 * np.pi * np.power(np.cos(x * np.pi), 2)) * ((x >= -0.5) * (x <= 0.5))