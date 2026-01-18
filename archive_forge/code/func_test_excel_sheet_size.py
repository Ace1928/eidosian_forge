from datetime import (
from functools import partial
from io import BytesIO
import os
import re
import numpy as np
import pytest
from pandas.compat import is_platform_windows
from pandas.compat._constants import PY310
from pandas.compat._optional import import_optional_dependency
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.excel import (
from pandas.io.excel._util import _writers
def test_excel_sheet_size(self, path):
    breaking_row_count = 2 ** 20 + 1
    breaking_col_count = 2 ** 14 + 1
    row_arr = np.zeros(shape=(breaking_row_count, 1))
    col_arr = np.zeros(shape=(1, breaking_col_count))
    row_df = DataFrame(row_arr)
    col_df = DataFrame(col_arr)
    msg = 'sheet is too large'
    with pytest.raises(ValueError, match=msg):
        row_df.to_excel(path)
    with pytest.raises(ValueError, match=msg):
        col_df.to_excel(path)