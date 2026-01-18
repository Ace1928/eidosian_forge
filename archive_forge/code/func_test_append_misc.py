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
def test_append_misc(setup_path):
    with ensure_clean_store(setup_path) as store:
        df = DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=Index(list('ABCD'), dtype=object), index=Index([f'i-{i}' for i in range(30)], dtype=object))
        store.append('df', df, chunksize=1)
        result = store.select('df')
        tm.assert_frame_equal(result, df)
        store.append('df1', df, expectedrows=10)
        result = store.select('df1')
        tm.assert_frame_equal(result, df)