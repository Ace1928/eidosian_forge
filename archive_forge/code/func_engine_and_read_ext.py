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
@pytest.fixture(params=[_transfer_marks(eng, ext) for eng in engine_params for ext in read_ext_params if _is_valid_engine_ext_pair(eng, ext)], ids=str)
def engine_and_read_ext(request):
    """
    Fixture for Excel reader engine and read_ext, only including valid pairs.
    """
    return request.param