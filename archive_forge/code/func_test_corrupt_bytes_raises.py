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
def test_corrupt_bytes_raises(self, engine):
    bad_stream = b'foo'
    if engine is None:
        error = ValueError
        msg = 'Excel file format cannot be determined, you must specify an engine manually.'
    elif engine == 'xlrd':
        from xlrd import XLRDError
        error = XLRDError
        msg = "Unsupported format, or corrupt file: Expected BOF record; found b'foo'"
    elif engine == 'calamine':
        from python_calamine import CalamineError
        error = CalamineError
        msg = 'Cannot detect file format'
    else:
        error = BadZipFile
        msg = 'File is not a zip file'
    with pytest.raises(error, match=msg):
        pd.read_excel(BytesIO(bad_stream))