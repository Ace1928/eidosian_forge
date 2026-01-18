from statsmodels.compat.python import lrange
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from scipy.linalg import toeplitz
from scipy.stats import t as student_t
from statsmodels.datasets import longley
from statsmodels.regression.linear_model import (
from statsmodels.tools.tools import add_constant
def test_ll(self):
    llf = np.array([r.llf for r in self.results])
    llf_1 = np.ones_like(llf) * self.results[0].llf
    assert_almost_equal(llf, llf_1, DECIMAL_7)
    ic = np.array([r.aic for r in self.results])
    ic_1 = np.ones_like(ic) * self.results[0].aic
    assert_almost_equal(ic, ic_1, DECIMAL_7)
    ic = np.array([r.bic for r in self.results])
    ic_1 = np.ones_like(ic) * self.results[0].bic
    assert_almost_equal(ic, ic_1, DECIMAL_7)