import math
import numpy as np
from numpy.testing import assert_allclose, assert_, assert_array_equal
from scipy.optimize import fmin_cobyla, minimize, Bounds
def c1(w):
    return 500 - abs(p(w)).sum()