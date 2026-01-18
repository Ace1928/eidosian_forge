from datetime import timezone
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_align_int(self, int_frame):
    other = DataFrame(index=range(5), columns=['A', 'B', 'C'])
    msg = "The 'method', 'limit', and 'fill_axis' keywords in DataFrame.align are deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        af, bf = int_frame.align(other, join='inner', axis=1, method='pad')
    tm.assert_index_equal(bf.columns, other.columns)