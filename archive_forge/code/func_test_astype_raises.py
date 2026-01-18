from datetime import datetime
import dateutil
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', [float, 'timedelta64', 'timedelta64[ns]', 'datetime64', 'datetime64[D]'])
def test_astype_raises(self, dtype):
    idx = DatetimeIndex(['2016-05-16', 'NaT', NaT, np.nan])
    msg = 'Cannot cast DatetimeIndex to dtype'
    if dtype == 'datetime64':
        msg = "Casting to unit-less dtype 'datetime64' is not supported"
    with pytest.raises(TypeError, match=msg):
        idx.astype(dtype)