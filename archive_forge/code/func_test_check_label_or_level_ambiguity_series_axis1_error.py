import pytest
from pandas.core.dtypes.missing import array_equivalent
import pandas as pd
def test_check_label_or_level_ambiguity_series_axis1_error(df):
    s = df.set_index('L1').L2
    with pytest.raises(ValueError, match='No axis named 1'):
        s._check_label_or_level_ambiguity('L1', axis=1)