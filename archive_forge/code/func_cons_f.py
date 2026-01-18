import pytest
import numpy as np
from numpy.testing import TestCase, assert_array_equal
import scipy.sparse as sps
from scipy.optimize._constraints import (
def cons_f(x):
    return np.array([x[0] ** 2 + x[1], x[0] ** 2 - x[1]])