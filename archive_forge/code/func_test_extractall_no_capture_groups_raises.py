from datetime import datetime
import re
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas import (
def test_extractall_no_capture_groups_raises(any_string_dtype):
    s = Series(['a3', 'b3', 'd4c2'], name='series_name', dtype=any_string_dtype)
    with pytest.raises(ValueError, match='no capture groups'):
        s.str.extractall('[a-z]')