import contextlib
from pathlib import Path
import re
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.io.excel import (
from pandas.io.excel._openpyxl import OpenpyxlReader
@pytest.mark.parametrize('kwarg_name', ['read_only', 'data_only'])
@pytest.mark.parametrize('kwarg_value', [True, False])
def test_engine_kwargs_append_reader(datapath, ext, kwarg_name, kwarg_value):
    filename = datapath('io', 'data', 'excel', 'test1' + ext)
    with contextlib.closing(OpenpyxlReader(filename, engine_kwargs={kwarg_name: kwarg_value})) as reader:
        assert getattr(reader.book, kwarg_name) == kwarg_value