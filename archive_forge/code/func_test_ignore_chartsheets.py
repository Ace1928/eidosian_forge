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
def test_ignore_chartsheets(self, request, engine, read_ext):
    if read_ext == '.ods':
        pytest.skip('chartsheets do not exist in the ODF format')
    if engine == 'pyxlsb':
        request.applymarker(pytest.mark.xfail(reason="pyxlsb can't distinguish chartsheets from worksheets"))
    with pd.ExcelFile('chartsheet' + read_ext) as excel:
        assert excel.sheet_names == ['Sheet1']