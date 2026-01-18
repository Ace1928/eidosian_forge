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
def test_exception_message_includes_sheet_name(self, read_ext):
    with pytest.raises(ValueError, match=' \\(sheet: Sheet1\\)$'):
        pd.read_excel('blank_with_header' + read_ext, header=[1], sheet_name=None)
    with pytest.raises(ZeroDivisionError, match=' \\(sheet: Sheet1\\)$'):
        pd.read_excel('test1' + read_ext, usecols=lambda x: 1 / 0, sheet_name=None)