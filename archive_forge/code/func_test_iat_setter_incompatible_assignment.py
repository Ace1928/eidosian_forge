from datetime import (
import itertools
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_iat_setter_incompatible_assignment(self):
    result = DataFrame({'a': [0.0, 1.0], 'b': [4, 5]})
    result.iat[0, 0] = None
    expected = DataFrame({'a': [None, 1], 'b': [4, 5]})
    tm.assert_frame_equal(result, expected)