import os
import pytest
from pandas import (
import pandas._testing as tm
def test_datapath(datapath):
    args = ('io', 'data', 'csv', 'iris.csv')
    result = datapath(*args)
    expected = os.path.join(os.path.dirname(os.path.dirname(__file__)), *args)
    assert result == expected