import numpy as np
from . import Filter  # prevent circular import in Python < 3.5
def gx(x, a):
    y = fx(x, a)
    return y / (y + fx(1 - x, a))