from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_nested_dict_overlapping_keys_replace_str(self):
    a = np.arange(1, 5)
    astr = a.astype(str)
    bstr = np.arange(2, 6).astype(str)
    df = DataFrame({'a': astr})
    result = df.replace(dict(zip(astr, bstr)))
    expected = df.replace({'a': dict(zip(astr, bstr))})
    tm.assert_frame_equal(result, expected)