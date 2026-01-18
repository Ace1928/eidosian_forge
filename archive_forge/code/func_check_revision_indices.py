from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import (
def check_revision_indices(news, revisions_index):
    if news.news_results.revision_impacts is None:
        assert_equal(len(news.revisions_iloc['revision date']), 0)
        assert_equal(len(news.revisions_iloc['revised variable']), 0)
        assert_equal(len(news.revisions_ix['revision date']), 0)
        assert_equal(len(news.revisions_ix['revised variable']), 0)
    else:
        dates = news.previous.model._index
        endog_names = news.previous.model.endog_names
        if isinstance(endog_names, str):
            endog_names = [endog_names]
        desired_ix = revisions_index.to_frame().reset_index(drop=True)
        desired_iloc = desired_ix.copy()
        desired_iloc['revision date'] = [dates.get_loc(date) for date in desired_ix['revision date']]
        desired_iloc['revised variable'] = [endog_names.index(name) for name in desired_ix['revised variable']]
        assert_(news.revisions_iloc.equals(desired_iloc.astype(news.revisions_iloc.dtypes)))
        assert_(news.revisions_ix.equals(desired_ix))