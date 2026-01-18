import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_datetime_frame_shift_with_freq_error(self, datetime_frame, frame_or_series):
    dtobj = tm.get_obj(datetime_frame, frame_or_series)
    no_freq = dtobj.iloc[[0, 5, 7]]
    msg = 'Freq was not set in the index hence cannot be inferred'
    with pytest.raises(ValueError, match=msg):
        no_freq.shift(freq='infer')