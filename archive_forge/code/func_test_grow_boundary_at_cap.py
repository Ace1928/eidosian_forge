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
@pytest.mark.slow
@pytest.mark.parametrize('count', [3 * 2 ** n for n in range(6)])
def test_grow_boundary_at_cap(c_parser_only, count):
    parser = c_parser_only
    with StringIO(',' * count) as s:
        expected = DataFrame(columns=[f'Unnamed: {i}' for i in range(count + 1)])
        df = parser.read_csv(s)
    tm.assert_frame_equal(df, expected)