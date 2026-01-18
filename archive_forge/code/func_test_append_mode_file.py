import contextlib
from pathlib import Path
import re
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.io.excel import (
from pandas.io.excel._openpyxl import OpenpyxlReader
def test_append_mode_file(ext):
    df = DataFrame()
    with tm.ensure_clean(ext) as f:
        df.to_excel(f, engine='openpyxl')
        with ExcelWriter(f, mode='a', engine='openpyxl', if_sheet_exists='new') as writer:
            df.to_excel(writer)
        data = Path(f).read_bytes()
        first = data.find(b'docProps/app.xml')
        second = data.find(b'docProps/app.xml', first + 1)
        third = data.find(b'docProps/app.xml', second + 1)
        assert second != -1 and third == -1