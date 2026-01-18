import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad_vec
from multiprocessing.dummy import Pool
def f_inf(x):
    return np.inf if x < 0.1 else 1 / x