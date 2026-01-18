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
def test_llnull(self):
    res = self.model.fit(start_params=self.start_params, disp=0)
    res._results._attach_nullmodel = True
    llf0 = res.llnull
    res_null0 = res.res_null
    assert_allclose(llf0, res_null0.llf, rtol=1e-06)
    res_null1 = self.res_null
    assert_allclose(llf0, res_null1.llf, rtol=1e-06)
    assert_allclose(res_null0.params, res_null1.params, rtol=5e-05)