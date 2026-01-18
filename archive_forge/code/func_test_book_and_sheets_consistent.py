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
def test_book_and_sheets_consistent(ext):
    with tm.ensure_clean(ext) as f:
        with ExcelWriter(f, engine='openpyxl') as writer:
            assert writer.sheets == {}
            sheet = writer.book.create_sheet('test_name', 0)
            assert writer.sheets == {'test_name': sheet}