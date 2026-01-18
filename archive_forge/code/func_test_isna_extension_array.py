import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
from pandas.core.dtypes.common import is_extension_array_dtype
from pandas.core.dtypes.dtypes import ExtensionDtype
import pandas as pd
import pandas._testing as tm
def test_isna_extension_array(self, data_missing):
    na = data_missing.isna()
    if is_extension_array_dtype(na):
        assert na._reduce('any')
        assert na.any()
        assert not na._reduce('all')
        assert not na.all()
        assert na.dtype._is_boolean