from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import (
@pytest.mark.parametrize('revisions', [True, False])
@pytest.mark.parametrize('updates', [True, False])
def test_dynamic_factor_time_varying(revisions, updates):
    endog = dta[['realgdp', 'unemp']].copy()
    endog['realgdp'] = np.log(endog['realgdp']).diff() * 400
    endog = endog.iloc[1:]
    comparison_type = None
    if updates:
        endog1 = endog.loc[:'2009Q2'].copy()
        endog2 = endog.loc[:'2009Q3'].copy()
    else:
        endog1 = endog.loc[:'2009Q3'].copy()
        endog2 = endog.loc[:'2009Q3'].copy()
        comparison_type = 'updated'
    if revisions:
        endog1.iloc[-1] = 0.0
    exog1 = np.ones_like(endog1['realgdp'])
    exog2 = np.ones_like(endog2['realgdp'])
    params1 = np.r_[0.9, 0.2, 0.0, 0.0, 1.2, 1.1, 0.5, 0.2]
    params2 = np.r_[0.9, 0.2, 1.2, 1.1, 0.5, 0.2]
    mod1 = dynamic_factor.DynamicFactor(endog1, exog=exog1, k_factors=1, factor_order=2)
    res1 = mod1.smooth(params1)
    news1 = res1.news(endog2, exog=exog2, start='2008Q1', end='2009Q3', comparison_type=comparison_type)
    mod2 = dynamic_factor.DynamicFactor(endog1, k_factors=1, factor_order=2)
    res2 = mod2.smooth(params2)
    news2 = res2.news(endog2, start='2008Q1', end='2009Q3', comparison_type=comparison_type)
    attrs = ['total_impacts', 'update_impacts', 'revision_impacts', 'news', 'weights', 'update_forecasts', 'update_realized', 'prev_impacted_forecasts', 'post_impacted_forecasts', 'revisions_iloc', 'revisions_ix', 'updates_iloc', 'updates_ix']
    for attr in attrs:
        w = getattr(news1, attr)
        x = getattr(news2, attr)
        if isinstance(x, pd.Series):
            assert_series_equal(w, x)
        else:
            assert_frame_equal(w, x)