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
def test_margeff_dummy(self):
    data = self.data
    vote = data.data['vote']
    exog = np.column_stack((data.exog, vote))
    exog = sm.add_constant(exog, prepend=False)
    res = MNLogit(data.endog, exog).fit(method='newton', disp=0)
    me = res.get_margeff(dummy=True)
    assert_almost_equal(me.margeff, self.res2.margeff_dydx_dummy_overall, 6)
    assert_almost_equal(me.margeff_se, self.res2.margeff_dydx_dummy_overall_se, 6)
    me = res.get_margeff(dummy=True, method='eydx')
    assert_almost_equal(me.margeff, self.res2.margeff_eydx_dummy_overall, 5)
    assert_almost_equal(me.margeff_se, self.res2.margeff_eydx_dummy_overall_se, 6)