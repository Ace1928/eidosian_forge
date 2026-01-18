import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod
@pytest.mark.parametrize('bins', [6, 7])
@pytest.mark.parametrize('box, compare', [(Series, tm.assert_series_equal), (np.array, tm.assert_categorical_equal), (list, tm.assert_equal)])
def test_cut_bool_coercion_to_int(bins, box, compare):
    data_expected = box([0, 1, 1, 0, 1] * 10)
    data_result = box([False, True, True, False, True] * 10)
    expected = cut(data_expected, bins, duplicates='drop')
    result = cut(data_result, bins, duplicates='drop')
    compare(result, expected)