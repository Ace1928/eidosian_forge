from datetime import timezone
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_align_mixed_type(self, float_string_frame):
    msg = "The 'method', 'limit', and 'fill_axis' keywords in DataFrame.align are deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        af, bf = float_string_frame.align(float_string_frame, join='inner', axis=1, method='pad')
    tm.assert_index_equal(bf.columns, float_string_frame.columns)