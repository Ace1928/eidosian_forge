from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@xfail_pyarrow
@pytest.mark.parametrize('na_filter', [True, False])
def test_infinity_parsing(all_parsers, na_filter):
    parser = all_parsers
    data = ',A\na,Infinity\nb,-Infinity\nc,+Infinity\n'
    expected = DataFrame({'A': [float('infinity'), float('-infinity'), float('+infinity')]}, index=['a', 'b', 'c'])
    result = parser.read_csv(StringIO(data), index_col=0, na_filter=na_filter)
    tm.assert_frame_equal(result, expected)