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
def test_custom_lineterminator(c_parser_only):
    parser = c_parser_only
    data = 'a,b,c~1,2,3~4,5,6'
    result = parser.read_csv(StringIO(data), lineterminator='~')
    expected = parser.read_csv(StringIO(data.replace('~', '\n')))
    tm.assert_frame_equal(result, expected)