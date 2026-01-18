from datetime import timedelta
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
from pandas import (
import pandas._testing as tm
def test_frozenset_index():
    idx0, idx1 = (frozenset('a'), frozenset('b'))
    s = Series([1, 2], index=[idx0, idx1])
    assert s[idx0] == 1
    assert s[idx1] == 2
    s[idx1] = 3
    assert s[idx1] == 3