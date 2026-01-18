from datetime import datetime
import re
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas import (
def test_extract_expand_True_mixed_object():
    er = [np.nan, np.nan]
    mixed = Series(['aBAD_BAD', np.nan, 'BAD_b_BAD', True, datetime.today(), 'foo', None, 1, 2.0])
    result = mixed.str.extract('.*(BAD[_]+).*(BAD)', expand=True)
    expected = DataFrame([['BAD_', 'BAD'], er, ['BAD_', 'BAD'], er, er, er, er, er, er], dtype=object)
    tm.assert_frame_equal(result, expected)