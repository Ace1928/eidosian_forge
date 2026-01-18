from statsmodels.compat.pandas import assert_index_equal
import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from scipy import stats
from scipy.stats import nbinom
import statsmodels.api as sm
from statsmodels.discrete.discrete_margins import _iscount, _isdummy
from statsmodels.discrete.discrete_model import (
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import (
from .results.results_discrete import Anes, DiscreteL1, RandHIE, Spector
def check_distr(res):
    distr = res.get_distribution()
    distr1 = res.model.get_distribution(res.params)
    m = res.predict()
    m2 = distr.mean()
    assert_allclose(m, np.squeeze(m2), rtol=1e-10)
    m2 = distr1.mean()
    assert_allclose(m, np.squeeze(m2), rtol=1e-10)
    v = res.predict(which='var')
    v2 = distr.var()
    assert_allclose(v, np.squeeze(v2), rtol=1e-10)