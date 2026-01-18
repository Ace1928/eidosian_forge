from io import (
import numpy as np
import pytest
import pandas._libs.parsers as parser
from pandas._libs.parsers import TextReader
from pandas.errors import ParserWarning
from pandas import DataFrame
import pandas._testing as tm
from pandas.io.parsers import (
from pandas.io.parsers.c_parser_wrapper import ensure_dtype_objs
def test_integer_thousands(self):
    data = '123,456\n12,500'
    reader = TextReader(StringIO(data), delimiter=':', thousands=',', header=None)
    result = reader.read()
    expected = np.array([123456, 12500], dtype=np.int64)
    tm.assert_almost_equal(result[0], expected)