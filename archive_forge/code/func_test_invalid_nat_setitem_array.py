from __future__ import annotations
import re
import warnings
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.parametrize('arr,non_casting_nats', [(TimedeltaIndex(['1 Day', '3 Hours', 'NaT'])._data, (np.datetime64('NaT', 'ns'), NaT._value)), (pd.date_range('2000-01-01', periods=3, freq='D')._data, (np.timedelta64('NaT', 'ns'), NaT._value)), (pd.period_range('2000-01-01', periods=3, freq='D')._data, (np.datetime64('NaT', 'ns'), np.timedelta64('NaT', 'ns'), NaT._value))], ids=lambda x: type(x).__name__)
def test_invalid_nat_setitem_array(arr, non_casting_nats):
    msg = "value should be a '(Timestamp|Timedelta|Period)', 'NaT', or array of those. Got '(timedelta64|datetime64|int)' instead."
    for nat in non_casting_nats:
        with pytest.raises(TypeError, match=msg):
            arr[0] = nat