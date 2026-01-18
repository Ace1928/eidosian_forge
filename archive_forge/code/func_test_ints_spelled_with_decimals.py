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
def test_ints_spelled_with_decimals(datapath, ext):
    path = datapath('io', 'data', 'excel', f'ints_spelled_with_decimals{ext}')
    result = pd.read_excel(path)
    expected = DataFrame(range(2, 12), columns=[1])
    tm.assert_frame_equal(result, expected)