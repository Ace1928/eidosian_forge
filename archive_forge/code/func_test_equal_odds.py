import os
import warnings
import numpy as np
import statsmodels.stats.contingency_tables as ctab
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
def test_equal_odds(self):
    if not hasattr(self, 'or_homog'):
        return
    rslt = self.rslt.test_equal_odds(adjust=False)
    assert_allclose(rslt.statistic, self.or_homog, rtol=0.0001, atol=0.0001)
    assert_allclose(rslt.pvalue, self.or_homog_p, rtol=0.0001, atol=0.0001)
    rslt = self.rslt.test_equal_odds(adjust=True)
    assert_allclose(rslt.statistic, self.or_homog_adj, rtol=0.0001, atol=0.0001)
    assert_allclose(rslt.pvalue, self.or_homog_adj_p, rtol=0.0001, atol=0.0001)