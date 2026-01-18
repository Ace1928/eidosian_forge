from io import StringIO
import os
import numpy as np
import pytest
from pandas._libs import parsers as libparsers
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_categorical_dtype_utf16(all_parsers, csv_dir_path):
    pth = os.path.join(csv_dir_path, 'utf16_ex.txt')
    parser = all_parsers
    encoding = 'utf-16'
    sep = '\t'
    expected = parser.read_csv(pth, sep=sep, encoding=encoding)
    expected = expected.apply(Categorical)
    actual = parser.read_csv(pth, sep=sep, encoding=encoding, dtype='category')
    tm.assert_frame_equal(actual, expected)