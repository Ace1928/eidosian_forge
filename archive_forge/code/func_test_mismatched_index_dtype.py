import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
def test_mismatched_index_dtype(self):
    left = pd.DataFrame([[to_datetime('20160602'), 1, 'a'], [to_datetime('20160602'), 2, 'a'], [to_datetime('20160603'), 1, 'b'], [to_datetime('20160603'), 2, 'b']], columns=['time', 'k1', 'k2']).set_index('time')
    left.index = left.index - pd.Timestamp(0)
    right = pd.DataFrame([[to_datetime('20160502'), 1, 'a', 1.0], [to_datetime('20160502'), 2, 'a', 2.0], [to_datetime('20160503'), 1, 'b', 3.0], [to_datetime('20160503'), 2, 'b', 4.0]], columns=['time', 'k1', 'k2', 'value']).set_index('time')
    msg = 'incompatible merge keys'
    with pytest.raises(MergeError, match=msg):
        merge_asof(left, right, left_index=True, right_index=True, by=['k1', 'k2'])