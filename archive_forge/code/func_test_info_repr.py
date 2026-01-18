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
def test_info_repr(self):
    term_width, term_height = get_terminal_size()
    max_rows = 60
    max_cols = 20 + (max(term_width, 80) - 80) // 4
    h, w = (max_rows + 1, max_cols - 1)
    df = DataFrame({k: np.arange(1, 1 + h) for k in np.arange(w)})
    assert has_vertically_truncated_repr(df)
    with option_context('display.large_repr', 'info'):
        assert has_info_repr(df)
    h, w = (max_rows - 1, max_cols + 1)
    df = DataFrame({k: np.arange(1, 1 + h) for k in np.arange(w)})
    assert has_horizontally_truncated_repr(df)
    with option_context('display.large_repr', 'info', 'display.max_columns', max_cols):
        assert has_info_repr(df)