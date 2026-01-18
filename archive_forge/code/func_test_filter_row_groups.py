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
def test_filter_row_groups(self, fp):
    d = {'a': list(range(3))}
    df = pd.DataFrame(d)
    with tm.ensure_clean() as path:
        df.to_parquet(path, engine=fp, compression=None, row_group_offsets=1)
        result = read_parquet(path, fp, filters=[('a', '==', 0)])
    assert len(result) == 1