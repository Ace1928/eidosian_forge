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
def test_columns_dtypes_not_invalid(self, pa):
    df = pd.DataFrame({'string': list('abc'), 'int': list(range(1, 4))})
    df.columns = [0, 1]
    check_round_trip(df, pa)
    df.columns = [b'foo', b'bar']
    with pytest.raises(NotImplementedError, match='|S3'):
        check_round_trip(df, pa)
    df.columns = [datetime.datetime(2011, 1, 1, 0, 0), datetime.datetime(2011, 1, 1, 1, 1)]
    check_round_trip(df, pa)