import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_intersection_equal_duplicates(self):
    idx = period_range('2011-01-01', periods=2)
    idx_dup = idx.append(idx)
    result = idx_dup.intersection(idx_dup)
    tm.assert_index_equal(result, idx)