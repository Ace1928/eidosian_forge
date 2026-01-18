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
@pytest.mark.parametrize('index_col', [None, 2])
def test_index_col_with_unnamed(self, read_ext, index_col):
    result = pd.read_excel('test1' + read_ext, sheet_name='Sheet4', index_col=index_col)
    expected = DataFrame([['i1', 'a', 'x'], ['i2', 'b', 'y']], columns=['Unnamed: 0', 'col1', 'col2'])
    if index_col:
        expected = expected.set_index(expected.columns[index_col])
    tm.assert_frame_equal(result, expected)