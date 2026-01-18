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
def test_expand_user(self, df_compat, monkeypatch):
    pytest.importorskip('pyarrow')
    monkeypatch.setenv('HOME', 'TestingUser')
    monkeypatch.setenv('USERPROFILE', 'TestingUser')
    with pytest.raises(OSError, match='.*TestingUser.*'):
        read_parquet('~/file.parquet')
    with pytest.raises(OSError, match='.*TestingUser.*'):
        df_compat.to_parquet('~/file.parquet')