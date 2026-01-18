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
def test_dietox(self):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    rdir = os.path.join(cur_dir, 'results')
    fname = os.path.join(rdir, 'dietox.csv')
    data = pd.read_csv(fname)
    model = MixedLM.from_formula('Weight ~ Time', groups='Pig', data=data)
    result = model.fit()
    assert_allclose(result.fe_params, np.r_[15.723523, 6.942505], rtol=1e-05)
    assert_allclose(result.bse[0:2], np.r_[0.78805374, 0.03338727], rtol=1e-05)
    assert_allclose(result.scale, 11.36692, rtol=1e-05)
    assert_allclose(result.cov_re, 40.39395, rtol=1e-05)
    assert_allclose(model.loglike(result.params_object), -2404.775, rtol=1e-05)
    data = pd.read_csv(fname)
    model = MixedLM.from_formula('Weight ~ Time', groups='Pig', data=data)
    result = model.fit(reml=False)
    assert_allclose(result.fe_params, np.r_[15.723517, 6.942506], rtol=1e-05)
    assert_allclose(result.bse[0:2], np.r_[0.7829397, 0.0333661], rtol=1e-05)
    assert_allclose(result.scale, 11.35251, rtol=1e-05)
    assert_allclose(result.cov_re, 39.82097, rtol=1e-05)
    assert_allclose(model.loglike(result.params_object), -2402.932, rtol=1e-05)