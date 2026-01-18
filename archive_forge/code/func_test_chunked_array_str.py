from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_chunked_array_str():
    data = [pa.array([1, 2, 3]), pa.array([4, 5, 6])]
    data = pa.chunked_array(data)
    assert str(data) == '[\n  [\n    1,\n    2,\n    3\n  ],\n  [\n    4,\n    5,\n    6\n  ]\n]'