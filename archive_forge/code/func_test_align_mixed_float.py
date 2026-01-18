from datetime import timezone
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_align_mixed_float(self, mixed_float_frame):
    other = DataFrame(index=range(5), columns=['A', 'B', 'C'])
    msg = "The 'method', 'limit', and 'fill_axis' keywords in DataFrame.align are deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        af, bf = mixed_float_frame.align(other.iloc[:, 0], join='inner', axis=1, method=None, fill_value=0)
    tm.assert_index_equal(bf.index, Index([]))