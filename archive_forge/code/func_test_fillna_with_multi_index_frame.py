import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import _check_mixed_float
def test_fillna_with_multi_index_frame(self):
    pdf = DataFrame({('x', 'a'): [np.nan, 2.0, 3.0], ('x', 'b'): [1.0, 2.0, np.nan], ('y', 'c'): [1.0, 2.0, np.nan]})
    expected = DataFrame({('x', 'a'): [-1.0, 2.0, 3.0], ('x', 'b'): [1.0, 2.0, -1.0], ('y', 'c'): [1.0, 2.0, np.nan]})
    tm.assert_frame_equal(pdf.fillna({'x': -1}), expected)
    tm.assert_frame_equal(pdf.fillna({'x': -1, ('x', 'b'): -2}), expected)
    expected = DataFrame({('x', 'a'): [-1.0, 2.0, 3.0], ('x', 'b'): [1.0, 2.0, -2.0], ('y', 'c'): [1.0, 2.0, np.nan]})
    tm.assert_frame_equal(pdf.fillna({('x', 'b'): -2, 'x': -1}), expected)