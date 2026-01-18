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
def check_external_error_on_write(self, df, engine, exc):
    with tm.ensure_clean() as path:
        with tm.external_error_raised(exc):
            to_parquet(df, path, engine, compression=None)