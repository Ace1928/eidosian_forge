from numpy.testing import (assert_allclose,
import pytest
import numpy as np
from scipy.optimize import direct, Bounds
def styblinski_tang(self, pos):
    x, y = pos
    return 0.5 * (x ** 4 - 16 * x ** 2 + 5 * x + y ** 4 - 16 * y ** 2 + 5 * y)