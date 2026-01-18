import datetime
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('convert_integer, expected', [(False, np.dtype('int32')), (True, 'Int32')])
def test_convert_dtypes(self, convert_integer, expected, string_storage, using_infer_string):
    if using_infer_string:
        string_storage = 'pyarrow_numpy'
    df = pd.DataFrame({'a': pd.Series([1, 2, 3], dtype=np.dtype('int32')), 'b': pd.Series(['x', 'y', 'z'], dtype=np.dtype('O'))})
    with pd.option_context('string_storage', string_storage):
        result = df.convert_dtypes(True, True, convert_integer, False)
    expected = pd.DataFrame({'a': pd.Series([1, 2, 3], dtype=expected), 'b': pd.Series(['x', 'y', 'z'], dtype=f'string[{string_storage}]')})
    tm.assert_frame_equal(result, expected)