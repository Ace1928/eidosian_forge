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
@pytest.mark.large_memory
def test_auto_chunking_list_of_binary():
    vals = [['x' * 1024]] * ((2 << 20) + 1)
    arr = pa.array(vals)
    assert isinstance(arr, pa.ChunkedArray)
    assert arr.num_chunks == 2
    assert len(arr.chunk(0)) == 2 ** 21 - 1
    assert len(arr.chunk(1)) == 2
    assert arr.chunk(1).to_pylist() == [['x' * 1024]] * 2