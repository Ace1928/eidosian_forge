import numpy as np
import pytest
from pandas import Series
import pandas._testing as tm
@pytest.fixture(params=[(1, 0), (5, 1)])
def rolling_consistency_cases(request):
    """window, min_periods"""
    return request.param