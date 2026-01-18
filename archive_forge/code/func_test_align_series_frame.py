import itertools
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import ExtensionArray
from pandas.core.internals.blocks import EABackedBlock
def test_align_series_frame(self, data, na_value):
    ser = pd.Series(data, name='a')
    df = pd.DataFrame({'col': np.arange(len(ser) + 1)})
    r1, r2 = ser.align(df)
    e1 = pd.Series(data._from_sequence(list(data) + [na_value], dtype=data.dtype), name=ser.name)
    tm.assert_series_equal(r1, e1)
    tm.assert_frame_equal(r2, df)