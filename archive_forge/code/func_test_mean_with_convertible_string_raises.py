import numpy as np
import pytest
import pandas as pd
from pandas import Series
import pandas._testing as tm
def test_mean_with_convertible_string_raises(using_array_manager, using_infer_string):
    ser = Series(['1', '2'])
    if using_infer_string:
        msg = 'does not support'
        with pytest.raises(TypeError, match=msg):
            ser.sum()
    else:
        assert ser.sum() == '12'
    msg = "Could not convert string '12' to numeric|does not support"
    with pytest.raises(TypeError, match=msg):
        ser.mean()
    df = ser.to_frame()
    if not using_array_manager:
        msg = "Could not convert \\['12'\\] to numeric|does not support"
    with pytest.raises(TypeError, match=msg):
        df.mean()