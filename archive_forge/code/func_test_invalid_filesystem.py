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
def test_invalid_filesystem(self):
    pytest.importorskip('pyarrow')
    df = pd.DataFrame(data={'A': [0, 1], 'B': [1, 0]})
    with tm.ensure_clean() as path:
        with pytest.raises(ValueError, match='filesystem must be a pyarrow or fsspec FileSystem'):
            df.to_parquet(path, engine='pyarrow', filesystem='foo')
    with tm.ensure_clean() as path:
        pathlib.Path(path).write_bytes(b'foo')
        with pytest.raises(ValueError, match='filesystem must be a pyarrow or fsspec FileSystem'):
            read_parquet(path, engine='pyarrow', filesystem='foo')