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
@pytest.mark.parametrize('sheet_name', [3, [0, 3], [3, 0], 'Sheet4', ['Sheet1', 'Sheet4'], ['Sheet4', 'Sheet1']])
def test_bad_sheetname_raises(self, read_ext, sheet_name):
    msg = "Worksheet index 3 is invalid|Worksheet named 'Sheet4' not found"
    with pytest.raises(ValueError, match=msg):
        with pd.ExcelFile('blank' + read_ext) as excel:
            excel.parse(sheet_name=sheet_name)