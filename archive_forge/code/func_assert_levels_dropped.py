import pytest
from pandas.core.dtypes.missing import array_equivalent
import pandas as pd
def assert_levels_dropped(frame, levels, axis):
    axis = frame._get_axis_number(axis)
    for level in levels:
        df_dropped = frame._drop_labels_or_levels(level, axis=axis)
        if axis == 0:
            assert level in frame.index.names
            assert level not in df_dropped.index.names
        else:
            assert level in frame.columns.names
            assert level not in df_dropped.columns.names