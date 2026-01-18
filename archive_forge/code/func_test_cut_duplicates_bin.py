import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod
@pytest.mark.parametrize('kwargs,msg', [({'duplicates': 'drop'}, None), ({}, 'Bin edges must be unique'), ({'duplicates': 'raise'}, 'Bin edges must be unique'), ({'duplicates': 'foo'}, "invalid value for 'duplicates' parameter")])
def test_cut_duplicates_bin(kwargs, msg):
    bins = [0, 2, 4, 6, 10, 10]
    values = Series(np.array([1, 3, 5, 7, 9]), index=['a', 'b', 'c', 'd', 'e'])
    if msg is not None:
        with pytest.raises(ValueError, match=msg):
            cut(values, bins, **kwargs)
    else:
        result = cut(values, bins, **kwargs)
        expected = cut(values, pd.unique(np.asarray(bins)))
        tm.assert_series_equal(result, expected)