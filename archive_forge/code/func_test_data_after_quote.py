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
def test_data_after_quote(c_parser_only):
    parser = c_parser_only
    data = 'a\n1\n"b"a'
    result = parser.read_csv(StringIO(data))
    expected = DataFrame({'a': ['1', 'ba']})
    tm.assert_frame_equal(result, expected)