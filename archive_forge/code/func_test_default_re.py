from statsmodels.compat.platform import PLATFORM_OSX
import os
import csv
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
import pytest
from statsmodels.regression.mixed_linear_model import (
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from statsmodels.base import _penalties as penalties
import statsmodels.tools.numdiff as nd
from .results import lme_r_results
def test_default_re(self):
    np.random.seed(3235)
    exog = np.random.normal(size=(300, 4))
    groups = np.kron(np.arange(100), [1, 1, 1])
    g_errors = np.kron(np.random.normal(size=100), [1, 1, 1])
    endog = exog.sum(1) + g_errors + np.random.normal(size=300)
    mdf1 = MixedLM(endog, exog, groups).fit()
    mdf2 = MixedLM(endog, exog, groups, np.ones(300)).fit()
    assert_almost_equal(mdf1.params, mdf2.params, decimal=8)