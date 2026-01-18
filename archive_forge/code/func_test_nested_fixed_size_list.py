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
@parametrize_with_sequence_types
def test_nested_fixed_size_list(seq):
    data = [[1, 2], [3, None], None]
    arr = pa.array(seq(data), type=pa.list_(pa.int64(), 2))
    assert len(arr) == 3
    assert arr.null_count == 1
    assert arr.type == pa.list_(pa.int64(), 2)
    assert arr.to_pylist() == data
    data = [np.array([1, 2], dtype='int64'), np.array([3, 4], dtype='int64'), None]
    arr = pa.array(seq(data), type=pa.list_(pa.int64(), 2))
    assert len(arr) == 3
    assert arr.null_count == 1
    assert arr.type == pa.list_(pa.int64(), 2)
    assert arr.to_pylist() == [[1, 2], [3, 4], None]
    data = [[1, 2, 4], [3, None], None]
    for data in [[[1, 2, 3]], [np.array([1, 2, 4], dtype='int64')]]:
        with pytest.raises(ValueError, match='Length of item not correct: expected 2'):
            pa.array(seq(data), type=pa.list_(pa.int64(), 2))
    data = [[], [], None]
    arr = pa.array(seq(data), type=pa.list_(pa.int64(), 0))
    assert len(arr) == 3
    assert arr.null_count == 1
    assert arr.type == pa.list_(pa.int64(), 0)
    assert arr.to_pylist() == [[], [], None]