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
@pytest.mark.parametrize('value_type', [pa.int8(), pa.int16(), pa.int32(), pa.int64(), pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64(), pa.float32(), pa.float64()])
def test_dictionary_from_integers(value_type):
    typ = pa.dictionary(pa.int8(), value_type=value_type)
    a = pa.array([1, 2, 1, 1, 2, 3], type=typ)
    assert isinstance(a.type, pa.DictionaryType)
    assert a.type.equals(typ)
    expected_indices = pa.array([0, 1, 0, 0, 1, 2], type=pa.int8())
    expected_dictionary = pa.array([1, 2, 3], type=value_type)
    assert a.indices.equals(expected_indices)
    assert a.dictionary.equals(expected_dictionary)