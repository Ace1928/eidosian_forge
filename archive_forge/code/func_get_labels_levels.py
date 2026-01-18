import pytest
from pandas.core.dtypes.missing import array_equivalent
import pandas as pd
def get_labels_levels(df_levels):
    expected_labels = list(df_levels.columns)
    expected_levels = [name for name in df_levels.index.names if name is not None]
    return (expected_labels, expected_levels)