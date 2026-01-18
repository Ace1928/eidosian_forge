import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_astype_tz_conversion(self):
    val = {'tz': date_range('2020-08-30', freq='d', periods=2, tz='Europe/London')}
    df = DataFrame(val)
    result = df.astype({'tz': 'datetime64[ns, Europe/Berlin]'})
    expected = df
    expected['tz'] = expected['tz'].dt.tz_convert('Europe/Berlin')
    tm.assert_frame_equal(result, expected)