import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_astype_category_readonly_mask_values(self):
    arr = array([0, 1, 2], dtype='Int64')
    arr._mask.flags['WRITEABLE'] = False
    result = arr.astype('category')
    expected = array([0, 1, 2], dtype='Int64').astype('category')
    tm.assert_extension_array_equal(result, expected)