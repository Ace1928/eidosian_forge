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
def test_map_from_tuples():
    expected = [[(b'a', 1), (b'b', 2)], [(b'c', 3)], [(b'd', 4), (b'e', 5), (b'f', None)], [(b'g', 7)]]
    arr = pa.array(expected, type=pa.map_(pa.binary(), pa.int32()))
    assert arr.to_pylist() == expected
    expected[1] = None
    arr = pa.array(expected, type=pa.map_(pa.binary(), pa.int32()))
    assert arr.to_pylist() == expected
    for entry in [[(5,)], [()], [('5', 'foo', True)]]:
        with pytest.raises(ValueError, match='(?i)tuple size'):
            pa.array([entry], type=pa.map_('i4', 'i4'))