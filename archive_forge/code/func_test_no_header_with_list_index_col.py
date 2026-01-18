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
def test_no_header_with_list_index_col(self, read_ext):
    file_name = 'testmultiindex' + read_ext
    data = [('B', 'B'), ('key', 'val'), (3, 4), (3, 4)]
    idx = MultiIndex.from_tuples([('A', 'A'), ('key', 'val'), (1, 2), (1, 2)], names=(0, 1))
    expected = DataFrame(data, index=idx, columns=(2, 3))
    result = pd.read_excel(file_name, sheet_name='index_col_none', index_col=[0, 1], header=None)
    tm.assert_frame_equal(expected, result)