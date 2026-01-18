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
def test_consistent_format(self):
    s = Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.9999, 1, 1] * 10)
    with option_context('display.max_rows', 10, 'display.show_dimensions', False):
        res = repr(s)
    exp = '0      1.0000\n1      1.0000\n2      1.0000\n3      1.0000\n4      1.0000\n        ...  \n125    1.0000\n126    1.0000\n127    0.9999\n128    1.0000\n129    1.0000\ndtype: float64'
    assert res == exp