import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_duplicated(self):
    df = tm.SubclassedDataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    result = df.duplicated()
    assert isinstance(result, tm.SubclassedSeries)
    df = tm.SubclassedDataFrame()
    result = df.duplicated()
    assert isinstance(result, tm.SubclassedSeries)