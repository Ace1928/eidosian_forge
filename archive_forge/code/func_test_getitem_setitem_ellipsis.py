from datetime import timedelta
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
from pandas import (
import pandas._testing as tm
def test_getitem_setitem_ellipsis(using_copy_on_write, warn_copy_on_write):
    s = Series(np.random.default_rng(2).standard_normal(10))
    result = s[...]
    tm.assert_series_equal(result, s)
    with tm.assert_cow_warning(warn_copy_on_write):
        s[...] = 5
    if not using_copy_on_write:
        assert (result == 5).all()