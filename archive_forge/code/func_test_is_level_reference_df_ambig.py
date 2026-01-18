import pytest
from pandas.core.dtypes.missing import array_equivalent
import pandas as pd
def test_is_level_reference_df_ambig(df_ambig, axis):
    axis = df_ambig._get_axis_number(axis)
    if axis == 1:
        df_ambig = df_ambig.T
    assert_label_reference(df_ambig, ['L1'], axis=axis)
    assert_level_reference(df_ambig, ['L2'], axis=axis)
    assert_label_reference(df_ambig, ['L3'], axis=axis)