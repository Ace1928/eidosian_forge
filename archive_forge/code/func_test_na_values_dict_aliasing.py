from io import StringIO
import numpy as np
import pytest
from pandas._libs.parsers import STR_NA_VALUES
from pandas import (
import pandas._testing as tm
def test_na_values_dict_aliasing(all_parsers):
    parser = all_parsers
    na_values = {'a': 2, 'b': 1}
    na_values_copy = na_values.copy()
    names = ['a', 'b']
    data = '1,2\n2,1'
    expected = DataFrame([[1.0, 2.0], [np.nan, np.nan]], columns=names)
    if parser.engine == 'pyarrow':
        msg = "The pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), names=names, na_values=na_values)
        return
    result = parser.read_csv(StringIO(data), names=names, na_values=na_values)
    tm.assert_frame_equal(result, expected)
    tm.assert_dict_equal(na_values, na_values_copy)