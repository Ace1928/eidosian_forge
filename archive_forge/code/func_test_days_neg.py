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
def test_days_neg(self):
    x = pd.to_timedelta(list(range(5)) + [NaT], unit='D')._values
    result = fmt._Timedelta64Formatter(-x).get_result()
    assert result[0].strip() == '0 days'
    assert result[1].strip() == '-1 days'