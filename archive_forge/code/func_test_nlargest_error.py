from string import ascii_lowercase
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.util.version import Version
@pytest.mark.parametrize('columns', [['group', 'category_string'], ['group', 'string']])
def test_nlargest_error(self, df_main_dtypes, nselect_method, columns):
    df = df_main_dtypes
    col = columns[1]
    error_msg = f"Column '{col}' has dtype {df[col].dtype}, cannot use method '{nselect_method}' with this dtype"
    error_msg = error_msg.replace('(', '\\(').replace(')', '\\)').replace('[', '\\[').replace(']', '\\]')
    with pytest.raises(TypeError, match=error_msg):
        getattr(df, nselect_method)(2, columns)