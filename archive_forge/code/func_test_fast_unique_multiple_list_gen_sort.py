import numpy as np
import pytest
from pandas._libs import (
from pandas.compat import IS64
from pandas import Index
import pandas._testing as tm
def test_fast_unique_multiple_list_gen_sort(self):
    keys = [['p', 'a'], ['n', 'd'], ['a', 's']]
    gen = (key for key in keys)
    expected = np.array(['a', 'd', 'n', 'p', 's'])
    out = lib.fast_unique_multiple_list_gen(gen, sort=True)
    tm.assert_numpy_array_equal(np.array(out), expected)
    gen = (key for key in keys)
    expected = np.array(['p', 'a', 'n', 'd', 's'])
    out = lib.fast_unique_multiple_list_gen(gen, sort=False)
    tm.assert_numpy_array_equal(np.array(out), expected)