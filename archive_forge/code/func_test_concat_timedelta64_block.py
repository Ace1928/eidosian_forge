import datetime as dt
from datetime import datetime
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_timedelta64_block():
    rng = to_timedelta(np.arange(10), unit='s')
    df = DataFrame({'time': rng})
    result = concat([df, df])
    tm.assert_frame_equal(result.iloc[:10], df)
    tm.assert_frame_equal(result.iloc[10:], df)