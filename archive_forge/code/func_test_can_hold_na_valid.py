import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
from pandas.core.dtypes.common import is_extension_array_dtype
from pandas.core.dtypes.dtypes import ExtensionDtype
import pandas as pd
import pandas._testing as tm
def test_can_hold_na_valid(self, data):
    assert data._can_hold_na is True