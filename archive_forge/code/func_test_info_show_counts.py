from io import StringIO
import re
from string import ascii_uppercase
import sys
import textwrap
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('row, columns, show_counts, result', [[20, 20, None, True], [20, 20, True, True], [20, 20, False, False], [5, 5, None, False], [5, 5, True, False], [5, 5, False, False]])
def test_info_show_counts(row, columns, show_counts, result):
    df = DataFrame(1, columns=range(10), index=range(10)).astype({1: 'float'})
    df.iloc[1, 1] = np.nan
    with option_context('display.max_info_rows', row, 'display.max_info_columns', columns):
        with StringIO() as buf:
            df.info(buf=buf, show_counts=show_counts)
            assert ('non-null' in buf.getvalue()) is result