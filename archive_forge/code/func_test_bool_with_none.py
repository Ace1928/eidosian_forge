import datetime
from decimal import Decimal
from io import BytesIO
import os
import pathlib
import numpy as np
import pytest
from pandas._config import using_copy_on_write
from pandas._config.config import _get_option
from pandas.compat import is_platform_windows
from pandas.compat.pyarrow import (
import pandas as pd
import pandas._testing as tm
from pandas.util.version import Version
from pandas.io.parquet import (
def test_bool_with_none(self, fp):
    df = pd.DataFrame({'a': [True, None, False]})
    expected = pd.DataFrame({'a': [1.0, np.nan, 0.0]}, dtype='float16')
    check_round_trip(df, fp, expected=expected, check_dtype=False)