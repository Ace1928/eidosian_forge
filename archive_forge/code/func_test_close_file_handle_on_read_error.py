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
def test_close_file_handle_on_read_error(self):
    with tm.ensure_clean('test.parquet') as path:
        pathlib.Path(path).write_bytes(b'breakit')
        with pytest.raises(Exception, match=''):
            read_parquet(path, engine='fastparquet')
        pathlib.Path(path).unlink(missing_ok=False)