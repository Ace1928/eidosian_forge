from datetime import (
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.frequencies import to_offset
def test_getitem_slice_keeps_name(self):
    st = Timestamp('2013-07-01 00:00:00', tz='America/Los_Angeles')
    et = Timestamp('2013-07-02 00:00:00', tz='America/Los_Angeles')
    dr = date_range(st, et, freq='h', name='timebucket')
    assert dr[1:].name == dr.name