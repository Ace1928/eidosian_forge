from datetime import (
import operator
import numpy as np
import pytest
from pandas import Timestamp
import pandas._testing as tm
def test_compare_invalid(self):
    val = Timestamp('20130101 12:01:02')
    assert not val == 'foo'
    assert not val == 10.0
    assert not val == 1
    assert not val == []
    assert not val == {'foo': 1}
    assert not val == np.float64(1)
    assert not val == np.int64(1)
    assert val != 'foo'
    assert val != 10.0
    assert val != 1
    assert val != []
    assert val != {'foo': 1}
    assert val != np.float64(1)
    assert val != np.int64(1)