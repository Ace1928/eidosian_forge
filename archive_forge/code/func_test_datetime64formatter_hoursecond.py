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
def test_datetime64formatter_hoursecond(self):
    x = Series(pd.to_datetime(['10:10:10.100', '12:12:12.120'], format='%H:%M:%S.%f'))._values

    def format_func(x):
        return x.strftime('%H:%M')
    formatter = fmt._Datetime64Formatter(x, formatter=format_func)
    result = formatter.get_result()
    assert result == ['10:10', '12:12']