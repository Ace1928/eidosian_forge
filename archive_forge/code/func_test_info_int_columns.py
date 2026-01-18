from io import StringIO
import re
from string import ascii_uppercase as uppercase
import sys
import textwrap
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
@pytest.mark.xfail(not IS64, reason='GH 36579: fail on 32-bit system')
def test_info_int_columns():
    df = DataFrame({1: [1, 2], 2: [2, 3]}, index=['A', 'B'])
    buf = StringIO()
    df.info(show_counts=True, buf=buf)
    result = buf.getvalue()
    expected = textwrap.dedent("        <class 'pandas.core.frame.DataFrame'>\n        Index: 2 entries, A to B\n        Data columns (total 2 columns):\n         #   Column  Non-Null Count  Dtype\n        ---  ------  --------------  -----\n         0   1       2 non-null      int64\n         1   2       2 non-null      int64\n        dtypes: int64(2)\n        memory usage: 48.0+ bytes\n        ")
    assert result == expected