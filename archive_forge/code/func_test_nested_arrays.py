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
def test_nested_arrays(seq):
    arr = pa.array(seq([np.array([], dtype=np.int64), np.array([1, 2], dtype=np.int64), None]))
    assert len(arr) == 3
    assert arr.null_count == 1
    assert arr.type == pa.list_(pa.int64())
    assert arr.to_pylist() == [[], [1, 2], None]