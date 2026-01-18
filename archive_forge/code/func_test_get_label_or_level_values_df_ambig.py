import pytest
from pandas.core.dtypes.missing import array_equivalent
import pandas as pd
def test_get_label_or_level_values_df_ambig(df_ambig, axis):
    axis = df_ambig._get_axis_number(axis)
    if axis == 1:
        df_ambig = df_ambig.T
    assert_level_values(df_ambig, ['L2'], axis=axis)
    assert_label_values(df_ambig, ['L3'], axis=axis)