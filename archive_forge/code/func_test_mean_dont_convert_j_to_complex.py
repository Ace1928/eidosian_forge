import numpy as np
import pytest
import pandas as pd
from pandas import Series
import pandas._testing as tm
def test_mean_dont_convert_j_to_complex(using_array_manager):
    df = pd.DataFrame([{'db': 'J', 'numeric': 123}])
    if using_array_manager:
        msg = "Could not convert string 'J' to numeric"
    else:
        msg = "Could not convert \\['J'\\] to numeric|does not support"
    with pytest.raises(TypeError, match=msg):
        df.mean()
    with pytest.raises(TypeError, match=msg):
        df.agg('mean')
    msg = "Could not convert string 'J' to numeric|does not support"
    with pytest.raises(TypeError, match=msg):
        df['db'].mean()
    msg = "Could not convert string 'J' to numeric|ufunc 'divide'"
    with pytest.raises(TypeError, match=msg):
        np.mean(df['db'].astype('string').array)