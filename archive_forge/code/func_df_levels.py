import pytest
from pandas.core.dtypes.missing import array_equivalent
import pandas as pd
@pytest.fixture(params=[[], ['L1'], ['L1', 'L2'], ['L1', 'L2', 'L3']])
def df_levels(request, df):
    """DataFrame with columns or index levels 'L1', 'L2', and 'L3'"""
    levels = request.param
    if levels:
        df = df.set_index(levels)
    return df