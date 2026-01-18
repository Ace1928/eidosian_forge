from itertools import product
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.forecasting.theta import ThetaModel
@pytest.fixture(params=['datetime', 'period', 'range', 'nofreq'])
def indexed_data(request):
    rs = np.random.RandomState([3290328901, 323293105, 121029109])
    scale = 0.01
    y = np.cumsum(scale + scale * rs.standard_normal(300))
    y = np.exp(y)
    if request.param == 'datetime':
        index = pd.date_range('2000-1-1', periods=300)
    elif request.param == 'period':
        index = pd.period_range('2000-1-1', periods=300, freq='M')
    elif request.param == 'range':
        index = pd.RangeIndex(100, 100 + 2 * 300, 2)
    else:
        index = pd.date_range('2000-1-1', periods=1000)
        locs = np.unique(rs.randint(0, 1000, size=500))
        index = index[locs[:300]]
    return pd.Series(y, index=index, name=f'y_{request.param}')