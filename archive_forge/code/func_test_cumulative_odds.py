import os
import warnings
import numpy as np
import statsmodels.stats.contingency_tables as ctab
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
def test_cumulative_odds():
    table = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    table = np.asarray(table)
    tbl_obj = ctab.Table(table)
    cum_odds = tbl_obj.cumulative_oddsratios
    assert_allclose(cum_odds[0, 0], 28 / float(5 * 11))
    assert_allclose(cum_odds[0, 1], 3 * 15 / float(3 * 24), atol=1e-05, rtol=1e-05)
    assert_allclose(np.log(cum_odds), tbl_obj.cumulative_log_oddsratios, atol=1e-05, rtol=1e-05)