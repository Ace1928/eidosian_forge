import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.arrays import BooleanArray
from pandas.core.ops.mask_ops import (
from pandas.tests.extension.base import BaseOpsUtil
def test_empty_ok(self, all_logical_operators):
    a = pd.array([], dtype='boolean')
    op_name = all_logical_operators
    result = getattr(a, op_name)(True)
    tm.assert_extension_array_equal(a, result)
    result = getattr(a, op_name)(False)
    tm.assert_extension_array_equal(a, result)
    result = getattr(a, op_name)(pd.NA)
    tm.assert_extension_array_equal(a, result)