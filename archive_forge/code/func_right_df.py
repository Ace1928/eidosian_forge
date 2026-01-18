import numpy as np
import pytest
from pandas import DataFrame
import pandas._testing as tm
@pytest.fixture(params=[[], ['outer'], ['outer', 'inner']])
def right_df(request, df2):
    """Construct right test DataFrame with specified levels
    (any of 'outer', 'inner', and 'v2')
    """
    levels = request.param
    if levels:
        df2 = df2.set_index(levels)
    return df2