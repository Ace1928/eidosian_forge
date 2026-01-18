from io import StringIO
import numpy as np
import pytest
from pandas._libs.parsers import STR_NA_VALUES
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('kwargs,expected', [({}, DataFrame({'A': ['a', 'b', np.nan, 'd', 'e', np.nan, 'g'], 'B': [1, 2, 3, 4, 5, 6, 7], 'C': ['one', 'two', 'three', np.nan, 'five', np.nan, 'seven']})), ({'na_values': {'A': [], 'C': []}, 'keep_default_na': False}, DataFrame({'A': ['a', 'b', '', 'd', 'e', 'nan', 'g'], 'B': [1, 2, 3, 4, 5, 6, 7], 'C': ['one', 'two', 'three', 'nan', 'five', '', 'seven']})), ({'na_values': ['a'], 'keep_default_na': False}, DataFrame({'A': [np.nan, 'b', '', 'd', 'e', 'nan', 'g'], 'B': [1, 2, 3, 4, 5, 6, 7], 'C': ['one', 'two', 'three', 'nan', 'five', '', 'seven']})), ({'na_values': {'A': [], 'C': []}}, DataFrame({'A': ['a', 'b', np.nan, 'd', 'e', np.nan, 'g'], 'B': [1, 2, 3, 4, 5, 6, 7], 'C': ['one', 'two', 'three', np.nan, 'five', np.nan, 'seven']}))])
def test_na_values_keep_default(all_parsers, kwargs, expected, request):
    data = 'A,B,C\na,1,one\nb,2,two\n,3,three\nd,4,nan\ne,5,five\nnan,6,\ng,7,seven\n'
    parser = all_parsers
    if parser.engine == 'pyarrow':
        if 'na_values' in kwargs and isinstance(kwargs['na_values'], dict):
            msg = "The pyarrow engine doesn't support passing a dict for na_values"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(StringIO(data), **kwargs)
            return
        mark = pytest.mark.xfail()
        request.applymarker(mark)
    result = parser.read_csv(StringIO(data), **kwargs)
    tm.assert_frame_equal(result, expected)