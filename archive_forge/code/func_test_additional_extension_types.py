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
def test_additional_extension_types(self, pa):
    pytest.importorskip('pyarrow')
    df = pd.DataFrame({'c': pd.IntervalIndex.from_tuples([(0, 1), (1, 2), (3, 4)]), 'd': pd.period_range('2012-01-01', periods=3, freq='D'), 'e': pd.IntervalIndex.from_breaks(pd.date_range('2012-01-01', periods=4, freq='D'))})
    check_round_trip(df, pa)