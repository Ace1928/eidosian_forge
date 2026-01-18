from functools import partial
import sys
import numpy as np
import pytest
import pandas._libs.window.aggregations as window_aggregations
from pandas import Series
import pandas._testing as tm
@pytest.fixture(params=_rolling_aggregations['params'], ids=_rolling_aggregations['ids'])
def rolling_aggregation(request):
    """Make a rolling aggregation function as fixture."""
    return request.param