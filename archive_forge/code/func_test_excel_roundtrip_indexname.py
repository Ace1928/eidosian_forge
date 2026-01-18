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
def test_excel_roundtrip_indexname(self, merge_cells, path):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
    df.index.name = 'foo'
    df.to_excel(path, merge_cells=merge_cells)
    with ExcelFile(path) as xf:
        result = pd.read_excel(xf, sheet_name=xf.sheet_names[0], index_col=0)
    tm.assert_frame_equal(result, df)
    assert result.index.name == 'foo'