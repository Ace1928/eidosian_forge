import numpy as np
import pytest
import pandas as pd
from pandas import Index
@pytest.fixture(params=[np.timedelta64(4, 'h'), pd.Timedelta(hours=23).to_pytimedelta(), pd.Timedelta('23:00:00')] + _common_mismatch)
def not_daily(request):
    """
    Several timedelta-like and DateOffset instances that are _not_
    compatible with Daily frequencies.
    """
    return request.param