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
@pytest.mark.parametrize('klass,ext', [pytest.param(_XlsxWriter, '.xlsx', marks=td.skip_if_no('xlsxwriter')), pytest.param(_OpenpyxlWriter, '.xlsx', marks=td.skip_if_no('openpyxl'))])
def test_ExcelWriter_dispatch(self, klass, ext):
    with tm.ensure_clean(ext) as path:
        with ExcelWriter(path) as writer:
            if ext == '.xlsx' and bool(import_optional_dependency('xlsxwriter', errors='ignore')):
                assert isinstance(writer, _XlsxWriter)
            else:
                assert isinstance(writer, klass)