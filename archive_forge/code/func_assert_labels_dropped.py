import pytest
from pandas.core.dtypes.missing import array_equivalent
import pandas as pd
def assert_labels_dropped(frame, labels, axis):
    axis = frame._get_axis_number(axis)
    for label in labels:
        df_dropped = frame._drop_labels_or_levels(label, axis=axis)
        if axis == 0:
            assert label in frame.columns
            assert label not in df_dropped.columns
        else:
            assert label in frame.index
            assert label not in df_dropped.index