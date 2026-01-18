from __future__ import annotations
from datetime import (
from functools import partial
from io import BytesIO
import os
from pathlib import Path
import platform
import re
from urllib.error import URLError
from zipfile import BadZipFile
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_excel_table_sheet_by_index(self, request, engine, read_ext, df_ref):
    xfail_datetimes_with_pyxlsb(engine, request)
    expected = df_ref
    adjust_expected(expected, read_ext, engine)
    with pd.ExcelFile('test1' + read_ext) as excel:
        df1 = pd.read_excel(excel, sheet_name=0, index_col=0)
        df2 = pd.read_excel(excel, sheet_name=1, skiprows=[1], index_col=0)
    tm.assert_frame_equal(df1, expected)
    tm.assert_frame_equal(df2, expected)
    with pd.ExcelFile('test1' + read_ext) as excel:
        df1 = excel.parse(0, index_col=0)
        df2 = excel.parse(1, skiprows=[1], index_col=0)
    tm.assert_frame_equal(df1, expected)
    tm.assert_frame_equal(df2, expected)
    with pd.ExcelFile('test1' + read_ext) as excel:
        df3 = pd.read_excel(excel, sheet_name=0, index_col=0, skipfooter=1)
    tm.assert_frame_equal(df3, df1.iloc[:-1])
    with pd.ExcelFile('test1' + read_ext) as excel:
        df3 = excel.parse(0, index_col=0, skipfooter=1)
    tm.assert_frame_equal(df3, df1.iloc[:-1])