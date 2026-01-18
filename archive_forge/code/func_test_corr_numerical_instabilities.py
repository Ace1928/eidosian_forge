import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_corr_numerical_instabilities(self):
    df = DataFrame([[0.2, 0.4], [0.4, 0.2]])
    result = df.corr()
    expected = DataFrame({0: [1.0, -1.0], 1: [-1.0, 1.0]})
    tm.assert_frame_equal(result - 1, expected - 1, atol=1e-17)