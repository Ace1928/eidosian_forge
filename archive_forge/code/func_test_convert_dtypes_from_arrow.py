import datetime
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_convert_dtypes_from_arrow(self):
    df = pd.DataFrame([['a', datetime.time(18, 12)]], columns=['a', 'b'])
    result = df.convert_dtypes()
    expected = df.astype({'a': 'string[python]'})
    tm.assert_frame_equal(result, expected)