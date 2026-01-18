from __future__ import annotations
from datetime import (
from functools import partial
from io import BytesIO
import os
from pathlib import Path
import platform
import re
from urllib.error import URLError
from zipfile import BadZipFile
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.fixture(autouse=True)
def cd_and_set_engine(self, engine, datapath, monkeypatch):
    """
        Change directory and set engine for ExcelFile objects.
        """
    func = partial(pd.ExcelFile, engine=engine)
    monkeypatch.chdir(datapath('io', 'data', 'excel'))
    monkeypatch.setattr(pd, 'ExcelFile', func)