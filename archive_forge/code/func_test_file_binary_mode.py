from decimal import Decimal
from io import (
import mmap
import os
import tarfile
import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p24
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_file_binary_mode(c_parser_only):
    parser = c_parser_only
    expected = DataFrame([[1, 2, 3], [4, 5, 6]])
    with tm.ensure_clean() as path:
        with open(path, 'w', encoding='utf-8') as f:
            f.write('1,2,3\n4,5,6')
        with open(path, 'rb') as f:
            result = parser.read_csv(f, header=None)
            tm.assert_frame_equal(result, expected)