import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_empty_dataframe_dtypes(self):
    df = DataFrame(columns=list('abc'))
    df['a'] = df['a'].astype(np.bool_)
    df['b'] = df['b'].astype(np.int32)
    df['c'] = df['c'].astype(np.float64)
    result = concat([df, df])
    assert result['a'].dtype == np.bool_
    assert result['b'].dtype == np.int32
    assert result['c'].dtype == np.float64
    result = concat([df, df.astype(np.float64)])
    assert result['a'].dtype == np.object_
    assert result['b'].dtype == np.float64
    assert result['c'].dtype == np.float64