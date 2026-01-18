from datetime import datetime
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.base import _registry as ea_registry
from pandas.core.dtypes.common import is_object_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tseries.offsets import BDay
def test_frame_setitem_newcol_timestamp(self):
    columns = date_range(start='1/1/2012', end='2/1/2012', freq=BDay())
    data = DataFrame(columns=columns, index=range(10))
    t = datetime(2012, 11, 1)
    ts = Timestamp(t)
    data[ts] = np.nan
    assert np.isnan(data[ts]).all()