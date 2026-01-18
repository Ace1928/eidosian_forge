from datetime import (
from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.io.formats.format as fmt
def test_masked_ea_with_formatter(self):
    df = DataFrame({'a': Series([0.123456789, 1.123456789], dtype='Float64'), 'b': Series([1, 2], dtype='Int64')})
    result = df.to_string(formatters=['{:.2f}'.format, '{:.2f}'.format])
    expected = '      a     b\n0  0.12  1.00\n1  1.12  2.00'
    assert result == expected