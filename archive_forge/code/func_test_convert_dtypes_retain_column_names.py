import datetime
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_convert_dtypes_retain_column_names(self):
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    df.columns.name = 'cols'
    result = df.convert_dtypes()
    tm.assert_index_equal(result.columns, df.columns)
    assert result.columns.name == 'cols'