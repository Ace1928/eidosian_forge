import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import ExtensionArray
from pandas.core.internals.blocks import EABackedBlock
def test_construct_empty_dataframe(self, dtype):
    result = pd.DataFrame(columns=['a'], dtype=dtype)
    expected = pd.DataFrame({'a': pd.array([], dtype=dtype)}, index=pd.RangeIndex(0))
    tm.assert_frame_equal(result, expected)