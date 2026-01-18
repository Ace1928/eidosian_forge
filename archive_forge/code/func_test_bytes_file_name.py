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
def test_bytes_file_name(self, engine):
    df = pd.DataFrame(data={'A': [0, 1], 'B': [1, 0]})
    with tm.ensure_clean('test.parquet') as path:
        with open(path.encode(), 'wb') as f:
            df.to_parquet(f)
        result = read_parquet(path, engine=engine)
    tm.assert_frame_equal(result, df)