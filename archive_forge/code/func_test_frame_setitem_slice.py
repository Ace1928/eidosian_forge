import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_frame_setitem_slice(multiindex_dataframe_random_data):
    df = multiindex_dataframe_random_data
    df.iloc[:4] = 0
    assert (df.values[:4] == 0).all()
    assert (df.values[4:] != 0).all()