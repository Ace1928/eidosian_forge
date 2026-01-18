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
def test_datetime64formatter_tz_ms(self):
    x = Series(np.array(['2999-01-01', '2999-01-02', 'NaT'], dtype='datetime64[ms]')).dt.tz_localize('US/Pacific')._values
    result = fmt._Datetime64TZFormatter(x).get_result()
    assert result[0].strip() == '2999-01-01 00:00:00-08:00'
    assert result[1].strip() == '2999-01-02 00:00:00-08:00'