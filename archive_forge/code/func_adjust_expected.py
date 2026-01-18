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
def adjust_expected(expected: DataFrame, read_ext: str, engine: str) -> None:
    expected.index.name = None
    unit = get_exp_unit(read_ext, engine)
    expected.index = expected.index.as_unit(unit)