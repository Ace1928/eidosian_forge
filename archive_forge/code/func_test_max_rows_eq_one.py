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
def test_max_rows_eq_one(self):
    s = Series(range(10), dtype='int64')
    with option_context('display.max_rows', 1):
        strrepr = repr(s).split('\n')
    exp1 = ['0', '0']
    res1 = strrepr[0].split()
    assert exp1 == res1
    exp2 = ['..']
    res2 = strrepr[1].split()
    assert exp2 == res2