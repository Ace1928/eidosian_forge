from io import StringIO
import numpy as np
import pytest
from pandas._libs.parsers import STR_NA_VALUES
from pandas import (
import pandas._testing as tm
def test_na_value_dict(all_parsers):
    data = 'A,B,C\nfoo,bar,NA\nbar,foo,foo\nfoo,bar,NA\nbar,foo,foo'
    parser = all_parsers
    if parser.engine == 'pyarrow':
        msg = "pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), na_values={'A': ['foo'], 'B': ['bar']})
        return
    df = parser.read_csv(StringIO(data), na_values={'A': ['foo'], 'B': ['bar']})
    expected = DataFrame({'A': [np.nan, 'bar', np.nan, 'bar'], 'B': [np.nan, 'foo', np.nan, 'foo'], 'C': [np.nan, 'foo', np.nan, 'foo']})
    tm.assert_frame_equal(df, expected)