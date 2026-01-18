import pytest
from numpy.testing import assert_allclose, assert_equal
import numpy as np
from math import pow
from scipy import optimize
def f38(self, x):
    return (100.0 * pow(x[1] - pow(x[0], 2), 2) + pow(1.0 - x[0], 2) + 90.0 * pow(x[3] - pow(x[2], 2), 2) + pow(1.0 - x[2], 2) + 10.1 * (pow(x[1] - 1.0, 2) + pow(x[3] - 1.0, 2)) + 19.8 * (x[1] - 1.0) * (x[3] - 1.0)) * 1e-05