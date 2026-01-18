from io import (
import os
import platform
from urllib.error import URLError
import uuid
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_internal_eof_byte(all_parsers):
    parser = all_parsers
    data = 'a,b\n1\x1a,2'
    expected = DataFrame([['1\x1a', 2]], columns=['a', 'b'])
    result = parser.read_csv(StringIO(data))
    tm.assert_frame_equal(result, expected)