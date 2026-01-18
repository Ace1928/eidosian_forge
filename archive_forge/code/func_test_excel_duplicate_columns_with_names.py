from datetime import (
from functools import partial
from io import BytesIO
import os
import re
import numpy as np
import pytest
from pandas.compat import is_platform_windows
from pandas.compat._constants import PY310
from pandas.compat._optional import import_optional_dependency
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.excel import (
from pandas.io.excel._util import _writers
def test_excel_duplicate_columns_with_names(self, path):
    df = DataFrame({'A': [0, 1], 'B': [10, 11]})
    df.to_excel(path, columns=['A', 'B', 'A'], index=False)
    result = pd.read_excel(path)
    expected = DataFrame([[0, 10, 0], [1, 11, 1]], columns=['A', 'B', 'A.1'])
    tm.assert_frame_equal(result, expected)