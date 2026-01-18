import collections
import datetime
import decimal
import itertools
import math
import re
import sys
import hypothesis as h
import numpy as np
import pytest
from pyarrow.pandas_compat import _pandas_api  # noqa
import pyarrow as pa
from pyarrow.tests import util
import pyarrow.tests.strategies as past
def test_convert_with_mask():
    data = [1, 2, 3, 4, 5]
    mask = np.array([False, True, False, False, True])
    result = pa.array(data, mask=mask)
    expected = pa.array([1, None, 3, 4, None])
    assert result.equals(expected)
    with pytest.raises(ValueError):
        pa.array(data, mask=mask[1:])