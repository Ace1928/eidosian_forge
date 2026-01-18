from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import (
def check_news_indices(news, updates_index, impact_dates):
    if len(updates_index):
        news_index = updates_index
    else:
        news_index = pd.MultiIndex.from_product([[], []], names=['update date', 'updated variable'])
    endog_names = news.previous.model.endog_names
    if isinstance(endog_names, str):
        endog_names = [endog_names]
    assert_(news.news.index.equals(news_index))
    assert_(news.update_forecasts.index.equals(news_index))
    assert_(news.update_realized.index.equals(news_index))
    assert_(news.weights.index.equals(news_index))
    weights_columns = pd.MultiIndex.from_product([impact_dates, endog_names])
    assert_(news.weights.columns.equals(weights_columns))