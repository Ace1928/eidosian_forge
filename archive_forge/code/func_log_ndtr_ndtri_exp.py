import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from scipy.special import log_ndtr, ndtri_exp
from scipy.special._testutils import assert_func_equal
def log_ndtr_ndtri_exp(y):
    return log_ndtr(ndtri_exp(y))