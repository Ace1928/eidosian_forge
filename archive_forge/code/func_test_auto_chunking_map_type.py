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
@pytest.mark.slow
@pytest.mark.large_memory
def test_auto_chunking_map_type():
    ty = pa.map_(pa.int8(), pa.int8())
    item = [(1, 1)] * 2 ** 28
    data = [item] * 2 ** 3
    arr = pa.array(data, type=ty)
    assert isinstance(arr, pa.ChunkedArray)
    assert len(arr.chunk(0)) == 7
    assert len(arr.chunk(1)) == 1