import numpy as np
import pytest
from pandas.core.dtypes import dtypes
from pandas.core.dtypes.common import is_extension_array_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import ExtensionArray
def test_ellipsis_index():
    df = pd.DataFrame({'col1': CapturingStringArray(np.array(['hello', 'world'], dtype=object))})
    _ = df.iloc[:1]
    out = df['col1'].array.last_item_arg
    assert str(out) == 'slice(None, 1, None)'