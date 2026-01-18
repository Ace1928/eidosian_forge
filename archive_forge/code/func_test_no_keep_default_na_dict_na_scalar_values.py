from io import StringIO
import numpy as np
import pytest
from pandas._libs.parsers import STR_NA_VALUES
from pandas import (
import pandas._testing as tm
def test_no_keep_default_na_dict_na_scalar_values(all_parsers):
    data = 'a,b\n1,2'
    parser = all_parsers
    if parser.engine == 'pyarrow':
        msg = "The pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), na_values={'b': 2}, keep_default_na=False)
        return
    df = parser.read_csv(StringIO(data), na_values={'b': 2}, keep_default_na=False)
    expected = DataFrame({'a': [1], 'b': [np.nan]})
    tm.assert_frame_equal(df, expected)