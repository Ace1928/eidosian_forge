import numpy as np
from . import Filter  # prevent circular import in Python < 3.5
def ffin(x, a):
    return gx(1 - x, a)