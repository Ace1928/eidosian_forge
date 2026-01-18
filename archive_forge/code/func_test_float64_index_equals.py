import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_float64_index_equals():
    float_index = Index([1.0, 2, 3])
    string_index = Index(['1', '2', '3'])
    result = float_index.equals(string_index)
    assert result is False
    result = string_index.equals(float_index)
    assert result is False