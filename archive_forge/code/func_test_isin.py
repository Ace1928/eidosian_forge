import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_isin(self):
    df = tm.SubclassedDataFrame({'num_legs': [2, 4], 'num_wings': [2, 0]}, index=['falcon', 'dog'])
    result = df.isin([0, 2])
    assert isinstance(result, tm.SubclassedDataFrame)