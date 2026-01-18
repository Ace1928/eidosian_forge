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
def test_limited_iterator_size_overflow():
    arr1 = pa.array(iter(range(3)), type=pa.int64(), size=2)
    arr2 = pa.array((0, 1))
    assert arr1.equals(arr2)