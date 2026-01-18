from datetime import datetime
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import MonthEnd
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_asfreq_resample_set_correct_freq(self, frame_or_series):
    dti = to_datetime(['2012-01-01', '2012-01-02', '2012-01-03'])
    obj = DataFrame({'col': [1, 2, 3]}, index=dti)
    obj = tm.get_obj(obj, frame_or_series)
    assert obj.index.freq is None
    assert obj.index.inferred_freq == 'D'
    assert obj.asfreq('D').index.freq == 'D'
    assert obj.resample('D').asfreq().index.freq == 'D'