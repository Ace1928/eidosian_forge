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
@pytest.mark.filterwarnings('ignore:Cell A4 is marked:UserWarning:openpyxl')
def test_date_conversion_overflow(self, request, engine, read_ext):
    xfail_datetimes_with_pyxlsb(engine, request)
    expected = DataFrame([[pd.Timestamp('2016-03-12'), 'Marc Johnson'], [pd.Timestamp('2016-03-16'), 'Jack Black'], [1e+20, 'Timothy Brown']], columns=['DateColWithBigInt', 'StringCol'])
    if engine == 'openpyxl':
        request.applymarker(pytest.mark.xfail(reason='Maybe not supported by openpyxl'))
    if engine is None and read_ext in ('.xlsx', '.xlsm'):
        request.applymarker(pytest.mark.xfail(reason='Defaults to openpyxl, maybe not supported'))
    result = pd.read_excel('testdateoverflow' + read_ext)
    tm.assert_frame_equal(result, expected)