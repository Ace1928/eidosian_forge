from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexing import IndexingError
from pandas.tseries.offsets import BDay
@pytest.mark.parametrize('tzstr', ['Europe/Berlin', 'dateutil/Europe/Berlin'])
def test_getitem_pydatetime_tz(self, tzstr):
    tz = timezones.maybe_get_tz(tzstr)
    index = date_range(start='2012-12-24 16:00', end='2012-12-24 18:00', freq='h', tz=tzstr)
    ts = Series(index=index, data=index.hour)
    time_pandas = Timestamp('2012-12-24 17:00', tz=tzstr)
    dt = datetime(2012, 12, 24, 17, 0)
    time_datetime = conversion.localize_pydatetime(dt, tz)
    assert ts[time_pandas] == ts[time_datetime]