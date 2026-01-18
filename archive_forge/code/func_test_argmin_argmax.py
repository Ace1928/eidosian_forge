import numpy as np
import pytest
from pandas.core.dtypes.common import is_bool_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.boolean import BooleanDtype
from pandas.tests.extension import base
def test_argmin_argmax(self, data_for_sorting, data_missing_for_sorting):
    assert data_for_sorting.argmax() == 0
    assert data_for_sorting.argmin() == 2
    data = data_for_sorting.take([2, 0, 0, 1, 1, 2])
    assert data.argmax() == 1
    assert data.argmin() == 0
    assert data_missing_for_sorting.argmax() == 0
    assert data_missing_for_sorting.argmin() == 2