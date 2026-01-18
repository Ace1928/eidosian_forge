import pytest
from pandas.core.dtypes.missing import array_equivalent
import pandas as pd
def test_is_level_reference_series_axis1_error(df):
    s = df.set_index('L1').L2
    with pytest.raises(ValueError, match='No axis named 1'):
        s._is_level_reference('L1', axis=1)