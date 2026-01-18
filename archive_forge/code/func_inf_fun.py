from numpy.testing import (assert_allclose,
import pytest
import numpy as np
from scipy.optimize import direct, Bounds
def inf_fun(self, x):
    return np.inf