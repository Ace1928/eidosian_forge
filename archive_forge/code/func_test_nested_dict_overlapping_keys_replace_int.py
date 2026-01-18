from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_nested_dict_overlapping_keys_replace_int(self):
    df = DataFrame({'a': list(range(1, 5))})
    result = df.replace({'a': dict(zip(range(1, 5), range(2, 6)))})
    expected = df.replace(dict(zip(range(1, 5), range(2, 6))))
    tm.assert_frame_equal(result, expected)