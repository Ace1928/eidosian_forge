from itertools import product
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.forecasting.theta import ThetaModel
def test_no_freq():
    idx = pd.date_range('2000-1-1', periods=300)
    locs = []
    for i in range(100):
        locs.append(2 * i + int(i % 2 == 1))
    y = pd.Series(np.random.standard_normal(100), index=idx[locs])
    with pytest.raises(ValueError, match='You must specify a period or'):
        ThetaModel(y)