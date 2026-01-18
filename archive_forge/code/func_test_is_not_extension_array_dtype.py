import numpy as np
import pytest
from pandas.core.dtypes import dtypes
from pandas.core.dtypes.common import is_extension_array_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import ExtensionArray
@pytest.mark.parametrize('values', [np.array([]), pd.Series(np.array([]))])
def test_is_not_extension_array_dtype(self, values):
    assert not is_extension_array_dtype(values)