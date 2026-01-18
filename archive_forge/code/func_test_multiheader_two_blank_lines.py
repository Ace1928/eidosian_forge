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
def test_multiheader_two_blank_lines(self, read_ext):
    file_name = 'testmultiindex' + read_ext
    columns = MultiIndex.from_tuples([('a', 'A'), ('b', 'B')])
    data = [[np.nan, np.nan], [np.nan, np.nan], [1, 3], [2, 4]]
    expected = DataFrame(data, columns=columns)
    result = pd.read_excel(file_name, sheet_name='mi_column_empty_rows', header=[0, 1])
    tm.assert_frame_equal(result, expected)