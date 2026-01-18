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
def test_comment_default(self, path):
    df = DataFrame({'A': ['one', '#one', 'one'], 'B': ['two', 'two', '#two']})
    df.to_excel(path, sheet_name='test_c')
    result1 = pd.read_excel(path, sheet_name='test_c')
    result2 = pd.read_excel(path, sheet_name='test_c', comment=None)
    tm.assert_frame_equal(result1, result2)