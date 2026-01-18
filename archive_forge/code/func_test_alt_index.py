from itertools import product
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.forecasting.theta import ThetaModel
@pytest.mark.smoke
def test_alt_index(indexed_data):
    idx = indexed_data.index
    date_like = not hasattr(idx, 'freq') or getattr(idx, 'freq', None) is None
    period = 12 if date_like else None
    res = ThetaModel(indexed_data, period=period).fit()
    if hasattr(idx, 'freq') and idx.freq is None:
        with pytest.warns(UserWarning):
            res.forecast_components(37)
        with pytest.warns(UserWarning):
            res.forecast(23)
    else:
        res.forecast_components(37)
        res.forecast(23)