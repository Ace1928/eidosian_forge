import numpy as np
import pytest
from pandas import DataFrame
import pandas._testing as tm
@pytest.fixture(params=[[], ['outer'], ['outer', 'inner']])
def left_df(request, df1):
    """Construct left test DataFrame with specified levels
    (any of 'outer', 'inner', and 'v1')
    """
    levels = request.param
    if levels:
        df1 = df1.set_index(levels)
    return df1