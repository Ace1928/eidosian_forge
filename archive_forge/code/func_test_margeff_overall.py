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
def test_margeff_overall(self):
    me = self.res1.get_margeff()
    assert_almost_equal(me.margeff, self.res2.margeff_dydx_overall, 6)
    assert_almost_equal(me.margeff_se, self.res2.margeff_dydx_overall_se, 6)
    me_frame = me.summary_frame()
    eff = me_frame['dy/dx'].values.reshape(me.margeff.shape, order='F')
    assert_allclose(eff, me.margeff, rtol=1e-13)
    assert_equal(me_frame.shape, (np.size(me.margeff), 6))