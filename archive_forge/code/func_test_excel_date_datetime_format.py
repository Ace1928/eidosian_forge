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
def test_excel_date_datetime_format(self, ext, path):
    unit = get_exp_unit(path)
    df = DataFrame([[date(2014, 1, 31), date(1999, 9, 24)], [datetime(1998, 5, 26, 23, 33, 4), datetime(2014, 2, 28, 13, 5, 13)]], index=['DATE', 'DATETIME'], columns=['X', 'Y'])
    df_expected = DataFrame([[datetime(2014, 1, 31), datetime(1999, 9, 24)], [datetime(1998, 5, 26, 23, 33, 4), datetime(2014, 2, 28, 13, 5, 13)]], index=['DATE', 'DATETIME'], columns=['X', 'Y'])
    df_expected = df_expected.astype(f'M8[{unit}]')
    with tm.ensure_clean(ext) as filename2:
        with ExcelWriter(path) as writer1:
            df.to_excel(writer1, sheet_name='test1')
        with ExcelWriter(filename2, date_format='DD.MM.YYYY', datetime_format='DD.MM.YYYY HH-MM-SS') as writer2:
            df.to_excel(writer2, sheet_name='test1')
        with ExcelFile(path) as reader1:
            rs1 = pd.read_excel(reader1, sheet_name='test1', index_col=0)
        with ExcelFile(filename2) as reader2:
            rs2 = pd.read_excel(reader2, sheet_name='test1', index_col=0)
    tm.assert_frame_equal(rs1, rs2)
    tm.assert_frame_equal(rs2, df_expected)