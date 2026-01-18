from copy import (
import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.compat.numpy import np_version_gte1p25
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_ndarray_compat_properties(index):
    if isinstance(index, PeriodIndex) and (not IS64):
        pytest.skip('Overflow')
    idx = index
    assert idx.T.equals(idx)
    assert idx.transpose().equals(idx)
    values = idx.values
    assert idx.shape == values.shape
    assert idx.ndim == values.ndim
    assert idx.size == values.size
    if not isinstance(index, (RangeIndex, MultiIndex)):
        assert idx.nbytes == values.nbytes
    idx.nbytes
    idx.values.nbytes