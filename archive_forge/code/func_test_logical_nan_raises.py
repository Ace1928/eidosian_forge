import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.arrays import BooleanArray
from pandas.core.ops.mask_ops import (
from pandas.tests.extension.base import BaseOpsUtil
def test_logical_nan_raises(self, all_logical_operators):
    op_name = all_logical_operators
    a = pd.array([True, False, None], dtype='boolean')
    msg = 'Got float instead'
    with pytest.raises(TypeError, match=msg):
        getattr(a, op_name)(np.nan)