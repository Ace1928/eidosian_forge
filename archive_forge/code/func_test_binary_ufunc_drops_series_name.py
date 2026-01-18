from collections import deque
import re
import string
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.arrays import SparseArray
def test_binary_ufunc_drops_series_name(ufunc, sparse, arrays_for_binary_ufunc):
    a1, a2 = arrays_for_binary_ufunc
    s1 = pd.Series(a1, name='a')
    s2 = pd.Series(a2, name='b')
    result = ufunc(s1, s2)
    assert result.name is None