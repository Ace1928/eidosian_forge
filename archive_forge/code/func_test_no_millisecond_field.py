import calendar
from datetime import datetime
import locale
import unicodedata
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
from pandas.tseries.frequencies import to_offset
def test_no_millisecond_field(self):
    msg = "type object 'DatetimeIndex' has no attribute 'millisecond'"
    with pytest.raises(AttributeError, match=msg):
        DatetimeIndex.millisecond
    msg = "'DatetimeIndex' object has no attribute 'millisecond'"
    with pytest.raises(AttributeError, match=msg):
        DatetimeIndex([]).millisecond