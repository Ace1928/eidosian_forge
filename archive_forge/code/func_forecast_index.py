from statsmodels.compat.pandas import (
from statsmodels.compat.pytest import pytest_warns
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.deterministic import (
@pytest.fixture(scope='module', params=[None, False, 'list'])
def forecast_index(request):
    idx = pd.date_range('2000-01-01', periods=400, freq='B')
    if request.param is None:
        return None
    elif request.param == 'list':
        return list(idx)
    return idx