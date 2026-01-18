import numpy as np
import pytest
import pandas as pd
from pandas.core.arrays.floating import (
def test_frame_repr(data_missing):
    df = pd.DataFrame({'A': data_missing})
    result = repr(df)
    expected = '      A\n0  <NA>\n1   0.1'
    assert result == expected