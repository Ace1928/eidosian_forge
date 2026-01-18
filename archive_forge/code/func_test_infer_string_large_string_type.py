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
def test_infer_string_large_string_type(self, tmp_path, pa):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = tmp_path / 'large_string.p'
    table = pa.table({'a': pa.array([None, 'b', 'c'], pa.large_string())})
    pq.write_table(table, path)
    with pd.option_context('future.infer_string', True):
        result = read_parquet(path)
    expected = pd.DataFrame(data={'a': [None, 'b', 'c']}, dtype='string[pyarrow_numpy]', columns=pd.Index(['a'], dtype='string[pyarrow_numpy]'))
    tm.assert_frame_equal(result, expected)