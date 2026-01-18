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
def test_excel_writer_context_manager(self, frame, path):
    with ExcelWriter(path) as writer:
        frame.to_excel(writer, sheet_name='Data1')
        frame2 = frame.copy()
        frame2.columns = frame.columns[::-1]
        frame2.to_excel(writer, sheet_name='Data2')
    with ExcelFile(path) as reader:
        found_df = pd.read_excel(reader, sheet_name='Data1', index_col=0)
        found_df2 = pd.read_excel(reader, sheet_name='Data2', index_col=0)
        tm.assert_frame_equal(found_df, frame)
        tm.assert_frame_equal(found_df2, frame2)