import pytest
from pandas.core.dtypes.missing import array_equivalent
import pandas as pd
def assert_label_values(frame, labels, axis):
    axis = frame._get_axis_number(axis)
    for label in labels:
        if axis == 0:
            expected = frame[label]._values
        else:
            expected = frame.loc[label]._values
        result = frame._get_label_or_level_values(label, axis=axis)
        assert array_equivalent(expected, result)