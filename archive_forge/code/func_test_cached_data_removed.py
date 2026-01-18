from statsmodels.compat.pandas import assert_series_equal
from io import BytesIO
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
def test_cached_data_removed(self):
    res = self.results
    names = ['resid_response', 'resid_deviance', 'resid_pearson', 'resid_anscombe']
    for name in names:
        getattr(res, name)
    for name in names:
        assert name in res._cache
        assert res._cache[name] is not None
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        res.remove_data()
    for name in names:
        assert res._cache[name] is None