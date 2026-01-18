from __future__ import division
import numpy as np
from pygsp import utils
from . import Filter  # prevent circular import in Python < 3.5
def low_pass(x):
    return np.exp(-x ** 4)