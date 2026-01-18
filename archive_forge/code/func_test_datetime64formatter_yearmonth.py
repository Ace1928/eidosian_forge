from datetime import datetime
from io import StringIO
from pathlib import Path
import re
from shutil import get_terminal_size
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
from pandas.io.formats import printing
import pandas.io.formats.format as fmt
def test_datetime64formatter_yearmonth(self):
    x = Series([datetime(2016, 1, 1), datetime(2016, 2, 2)])._values

    def format_func(x):
        return x.strftime('%Y-%m')
    formatter = fmt._Datetime64Formatter(x, formatter=format_func)
    result = formatter.get_result()
    assert result == ['2016-01', '2016-02']