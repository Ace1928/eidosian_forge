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
def test_map_from_dicts():
    data = [[{'key': b'a', 'value': 1}, {'key': b'b', 'value': 2}], [{'key': b'c', 'value': 3}], [{'key': b'd', 'value': 4}, {'key': b'e', 'value': 5}, {'key': b'f', 'value': None}], [{'key': b'g', 'value': 7}]]
    expected = [[(d['key'], d['value']) for d in entry] for entry in data]
    arr = pa.array(expected, type=pa.map_(pa.binary(), pa.int32()))
    assert arr.to_pylist() == expected
    data[1] = None
    expected[1] = None
    arr = pa.array(expected, type=pa.map_(pa.binary(), pa.int32()))
    assert arr.to_pylist() == expected
    for entry in [[{'value': 5}], [{}], [{'k': 1, 'v': 2}]]:
        with pytest.raises(ValueError, match='Invalid Map'):
            pa.array([entry], type=pa.map_('i4', 'i4'))
    for entry in [[{'key': '1', 'value': 5}], [{'key': {'value': 2}}]]:
        with pytest.raises(pa.ArrowInvalid, match='tried to convert to int'):
            pa.array([entry], type=pa.map_('i4', 'i4'))