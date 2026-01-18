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
def test_engine_kwargs_append_invalid(ext):
    with tm.ensure_clean(ext) as f:
        DataFrame(['hello', 'world']).to_excel(f)
        with pytest.raises(TypeError, match=re.escape("load_workbook() got an unexpected keyword argument 'apple_banana'")):
            with ExcelWriter(f, engine='openpyxl', mode='a', engine_kwargs={'apple_banana': 'fruit'}) as writer:
                DataFrame(['good']).to_excel(writer, sheet_name='Sheet2')