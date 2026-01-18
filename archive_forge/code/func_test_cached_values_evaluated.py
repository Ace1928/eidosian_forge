from statsmodels.compat.pandas import assert_series_equal
from io import BytesIO
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
def test_cached_values_evaluated(self):
    res = self.results
    assert res._cache == {}
    res.remove_data()
    assert 'aic' in res._cache