from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_loc_past_lexsort_depth(self):
    idx = MultiIndex(levels=[['a'], [0, 7], [1]], codes=[[0, 0], [1, 0], [0, 0]], names=['x', 'y', 'z'], sortorder=0)
    key = ('a', 7)
    with tm.assert_produces_warning(PerformanceWarning):
        result = idx.get_loc(key)
    assert result == slice(0, 1, None)