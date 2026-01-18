from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.tests import test_dynamic_factor_mq_monte_carlo
def test_news_monthly(reset_randomstate):
    endog, _, _, _, _, _ = gen_dfm_data(k_endog=10, nobs=100)
    endog_pre = endog.iloc[:-1].copy()
    endog_pre.iloc[-1, 0] *= 1.2
    endog_pre.iloc[-1, 1] = np.nan
    mod = dynamic_factor_mq.DynamicFactorMQ(endog_pre, factor_orders=1, standardize=False, idiosyncratic_ar1=False)
    params = mod.start_params
    res = mod.smooth(params)
    mod2 = mod.clone(endog)
    res2 = mod2.smooth(params)
    desired = res2.news(res, start=endog.index[-1], periods=1, comparison_type='previous')
    actual = res.news(endog, start=endog.index[-1], periods=1, comparison_type='updated')
    attributes = ['total_impacts', 'update_impacts', 'revision_impacts', 'news', 'weights', 'update_forecasts', 'update_realized', 'prev_impacted_forecasts', 'post_impacted_forecasts', 'revisions_iloc', 'revisions_ix', 'updates_iloc', 'updates_ix']
    for attr in attributes:
        w = getattr(actual, attr)
        x = getattr(desired, attr)
        if isinstance(x, pd.Series):
            assert_series_equal(w, x)
        else:
            assert_frame_equal(w, x)