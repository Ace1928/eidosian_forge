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
def test_freeze_panes(self, path):
    expected = DataFrame([[1, 2], [3, 4]], columns=['col1', 'col2'])
    expected.to_excel(path, sheet_name='Sheet1', freeze_panes=(1, 1))
    result = pd.read_excel(path, index_col=0)
    tm.assert_frame_equal(result, expected)