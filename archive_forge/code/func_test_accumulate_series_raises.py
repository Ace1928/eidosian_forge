import numpy as np
import pytest
from pandas.core.dtypes.common import is_bool_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.boolean import BooleanDtype
from pandas.tests.extension import base
@pytest.mark.parametrize('skipna', [True, False])
def test_accumulate_series_raises(self, data, all_numeric_accumulations, skipna):
    pass