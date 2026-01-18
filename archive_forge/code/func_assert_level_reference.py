import pytest
from pandas.core.dtypes.missing import array_equivalent
import pandas as pd
def assert_level_reference(frame, levels, axis):
    for level in levels:
        assert frame._is_level_reference(level, axis=axis)
        assert not frame._is_label_reference(level, axis=axis)
        assert frame._is_label_or_level_reference(level, axis=axis)