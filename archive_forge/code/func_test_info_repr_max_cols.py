from datetime import datetime
from io import StringIO
from pathlib import Path
import re
from shutil import get_terminal_size
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
from pandas.io.formats import printing
import pandas.io.formats.format as fmt
def test_info_repr_max_cols(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
    with option_context('display.large_repr', 'info', 'display.max_columns', 1, 'display.max_info_columns', 4):
        assert has_non_verbose_info_repr(df)
    with option_context('display.large_repr', 'info', 'display.max_columns', 1, 'display.max_info_columns', 5):
        assert not has_non_verbose_info_repr(df)