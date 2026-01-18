import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.arrays import BooleanArray
from pandas.core.ops.mask_ops import (
from pandas.tests.extension.base import BaseOpsUtil
@pytest.mark.parametrize('other', ['a', 1])
def test_non_bool_or_na_other_raises(self, other, all_logical_operators):
    a = pd.array([True, False], dtype='boolean')
    with pytest.raises(TypeError, match=str(type(other).__name__)):
        getattr(a, all_logical_operators)(other)