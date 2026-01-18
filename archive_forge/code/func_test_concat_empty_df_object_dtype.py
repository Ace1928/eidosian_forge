import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_empty_df_object_dtype(self):
    df_1 = DataFrame({'Row': [0, 1, 1], 'EmptyCol': np.nan, 'NumberCol': [1, 2, 3]})
    df_2 = DataFrame(columns=df_1.columns)
    result = concat([df_1, df_2], axis=0)
    expected = df_1.astype(object)
    tm.assert_frame_equal(result, expected)