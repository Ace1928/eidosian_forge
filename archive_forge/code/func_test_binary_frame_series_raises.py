from functools import partial
import re
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_extension_array_dtype
def test_binary_frame_series_raises():
    df = pd.DataFrame({'A': [1, 2]})
    with pytest.raises(NotImplementedError, match='logaddexp'):
        np.logaddexp(df, df['A'])
    with pytest.raises(NotImplementedError, match='logaddexp'):
        np.logaddexp(df['A'], df)