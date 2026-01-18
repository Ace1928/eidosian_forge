from io import StringIO
import numpy as np
import pytest
from pandas._libs.parsers import STR_NA_VALUES
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data,kwargs,expected', [(str(2 ** 63) + '\n' + str(2 ** 63 + 1), {'na_values': [2 ** 63]}, DataFrame([str(2 ** 63), str(2 ** 63 + 1)])), (str(2 ** 63) + ',1' + '\n,2', {}, DataFrame([[str(2 ** 63), 1], ['', 2]])), (str(2 ** 63) + '\n1', {'na_values': [2 ** 63]}, DataFrame([np.nan, 1]))])
def test_na_values_uint64(all_parsers, data, kwargs, expected, request):
    parser = all_parsers
    if parser.engine == 'pyarrow' and 'na_values' in kwargs:
        msg = "The 'pyarrow' engine requires all na_values to be strings"
        with pytest.raises(TypeError, match=msg):
            parser.read_csv(StringIO(data), header=None, **kwargs)
        return
    elif parser.engine == 'pyarrow':
        mark = pytest.mark.xfail(reason='Returns float64 instead of object')
        request.applymarker(mark)
    result = parser.read_csv(StringIO(data), header=None, **kwargs)
    tm.assert_frame_equal(result, expected)