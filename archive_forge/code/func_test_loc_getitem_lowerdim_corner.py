import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_getitem_lowerdim_corner(multiindex_dataframe_random_data):
    df = multiindex_dataframe_random_data
    with pytest.raises(KeyError, match="^\\('bar', 'three'\\)$"):
        df.loc[('bar', 'three'), 'B']
    df.loc[('bar', 'three'), 'B'] = 0
    expected = 0
    result = df.sort_index().loc[('bar', 'three'), 'B']
    assert result == expected