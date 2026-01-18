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
def test_basic_results(self):
    assert_almost_equal(self.res1.params, self.res2.params, DECIMAL_4)
    assert_almost_equal(self.res1.cov_params(), self.res2.cov_params(), DECIMAL_4)
    assert_almost_equal(self.res1.conf_int(), self.res2.conf_int(), DECIMAL_4)
    assert_almost_equal(self.res1.pvalues, self.res2.pvalues, DECIMAL_4)
    assert_almost_equal(self.res1.pred_table(), self.res2.pred_table(), DECIMAL_4)
    assert_almost_equal(self.res1.bse, self.res2.bse, DECIMAL_4)
    assert_almost_equal(self.res1.llf, self.res2.llf, DECIMAL_4)
    assert_almost_equal(self.res1.aic, self.res2.aic, DECIMAL_4)
    assert_almost_equal(self.res1.bic, self.res2.bic, DECIMAL_4)
    assert_almost_equal(self.res1.pvalues, self.res2.pvalues, DECIMAL_4)
    assert_(self.res1.mle_retvals['converged'] is True)