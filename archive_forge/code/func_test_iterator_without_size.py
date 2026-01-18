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
def test_iterator_without_size():
    expected = pa.array((0, 1, 2))
    arr1 = pa.array(iter(range(3)))
    assert arr1.equals(expected)
    arr1 = pa.array(iter(range(3)), type=pa.int64())
    assert arr1.equals(expected)