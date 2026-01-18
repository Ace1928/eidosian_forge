from collections import defaultdict
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_dtype_backend_ea_dtype_specified(all_parsers):
    data = 'a,b\n1,2\n'
    parser = all_parsers
    result = parser.read_csv(StringIO(data), dtype='Int64', dtype_backend='numpy_nullable')
    expected = DataFrame({'a': [1], 'b': 2}, dtype='Int64')
    tm.assert_frame_equal(result, expected)