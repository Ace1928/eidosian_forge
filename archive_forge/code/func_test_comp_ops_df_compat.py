from datetime import (
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import IncompatibleFrequency
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.core.computation.check import NUMEXPR_INSTALLED
@pytest.mark.parametrize('left, right', [(Series([1, 2, 3], index=list('ABC'), name='x'), Series([2, 2, 2], index=list('ABD'), name='x')), (Series([1, 2, 3], index=list('ABC'), name='x'), Series([2, 2, 2, 2], index=list('ABCD'), name='x'))])
def test_comp_ops_df_compat(self, left, right, frame_or_series):
    if frame_or_series is not Series:
        msg = f'Can only compare identically-labeled \\(both index and columns\\) {frame_or_series.__name__} objects'
        left = left.to_frame()
        right = right.to_frame()
    else:
        msg = f'Can only compare identically-labeled {frame_or_series.__name__} objects'
    with pytest.raises(ValueError, match=msg):
        left == right
    with pytest.raises(ValueError, match=msg):
        right == left
    with pytest.raises(ValueError, match=msg):
        left != right
    with pytest.raises(ValueError, match=msg):
        right != left
    with pytest.raises(ValueError, match=msg):
        left < right
    with pytest.raises(ValueError, match=msg):
        right < left