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
def test_deprecate_bytes_input(self, engine, read_ext):
    msg = "Passing bytes to 'read_excel' is deprecated and will be removed in a future version. To read from a byte string, wrap it in a `BytesIO` object."
    with tm.assert_produces_warning(FutureWarning, match=msg, raise_on_extra_warnings=False):
        with open('test1' + read_ext, 'rb') as f:
            pd.read_excel(f.read(), engine=engine)