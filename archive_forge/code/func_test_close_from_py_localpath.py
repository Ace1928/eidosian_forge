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
def test_close_from_py_localpath(self, read_ext):
    str_path = os.path.join('test1' + read_ext)
    with open(str_path, 'rb') as f:
        x = pd.read_excel(f, sheet_name='Sheet1', index_col=0)
        del x
        f.read()