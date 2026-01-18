from io import StringIO
import numpy as np
import pytest
from pandas._libs.parsers import STR_NA_VALUES
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('na_filter,index_data', [(False, ['', '5']), (True, [np.nan, 5.0])])
def test_no_na_filter_on_index(all_parsers, na_filter, index_data, request):
    parser = all_parsers
    data = 'a,b,c\n1,,3\n4,5,6'
    if parser.engine == 'pyarrow' and na_filter is False:
        mark = pytest.mark.xfail(reason='mismatched index result')
        request.applymarker(mark)
    expected = DataFrame({'a': [1, 4], 'c': [3, 6]}, index=Index(index_data, name='b'))
    result = parser.read_csv(StringIO(data), index_col=[1], na_filter=na_filter)
    tm.assert_frame_equal(result, expected)