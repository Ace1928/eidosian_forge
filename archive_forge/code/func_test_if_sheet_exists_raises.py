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
@pytest.mark.parametrize('if_sheet_exists,msg', [('invalid', "'invalid' is not valid for if_sheet_exists. Valid options are 'error', 'new', 'replace' and 'overlay'."), ('error', "Sheet 'foo' already exists and if_sheet_exists is set to 'error'."), (None, "Sheet 'foo' already exists and if_sheet_exists is set to 'error'.")])
def test_if_sheet_exists_raises(ext, if_sheet_exists, msg):
    df = DataFrame({'fruit': ['pear']})
    with tm.ensure_clean(ext) as f:
        with pytest.raises(ValueError, match=re.escape(msg)):
            df.to_excel(f, sheet_name='foo', engine='openpyxl')
            with ExcelWriter(f, engine='openpyxl', mode='a', if_sheet_exists=if_sheet_exists) as writer:
                df.to_excel(writer, sheet_name='foo')