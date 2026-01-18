import datetime
from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
@pytest.mark.parametrize('chunksize', [10, 200, 1000])
def test_append_misc_chunksize(setup_path, chunksize):
    df = DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=Index(list('ABCD'), dtype=object), index=Index([f'i-{i}' for i in range(30)], dtype=object))
    df['string'] = 'foo'
    df['float322'] = 1.0
    df['float322'] = df['float322'].astype('float32')
    df['bool'] = df['float322'] > 0
    df['time1'] = Timestamp('20130101').as_unit('ns')
    df['time2'] = Timestamp('20130102').as_unit('ns')
    with ensure_clean_store(setup_path, mode='w') as store:
        store.append('obj', df, chunksize=chunksize)
        result = store.select('obj')
        tm.assert_frame_equal(result, df)