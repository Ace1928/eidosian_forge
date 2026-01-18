import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import pytest
from scipy import stats
from scipy.stats import sobol_indices
from scipy.stats._resampling import BootstrapResult
from scipy.stats._sensitivity_analysis import (
def f_ishigami_vec_const(x):
    """Output of shape (3, n)."""
    res = f_ishigami(x)
    return (res, res * 0 + 10, res)