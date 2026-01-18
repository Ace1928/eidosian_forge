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
def test_creating_and_reading_multiple_sheets(self, ext):

    def tdf(col_sheet_name):
        d, i = ([11, 22, 33], [1, 2, 3])
        return DataFrame(d, i, columns=[col_sheet_name])
    sheets = ['AAA', 'BBB', 'CCC']
    dfs = [tdf(s) for s in sheets]
    dfs = dict(zip(sheets, dfs))
    with tm.ensure_clean(ext) as pth:
        with ExcelWriter(pth) as ew:
            for sheetname, df in dfs.items():
                df.to_excel(ew, sheet_name=sheetname)
        dfs_returned = pd.read_excel(pth, sheet_name=sheets, index_col=0)
        for s in sheets:
            tm.assert_frame_equal(dfs[s], dfs_returned[s])