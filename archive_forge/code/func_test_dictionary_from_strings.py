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
def test_dictionary_from_strings():
    for value_type in [pa.binary(), pa.string()]:
        typ = pa.dictionary(pa.int8(), value_type)
        a = pa.array(['', 'a', 'bb', 'a', 'bb', 'ccc'], type=typ)
        assert isinstance(a.type, pa.DictionaryType)
        expected_indices = pa.array([0, 1, 2, 1, 2, 3], type=pa.int8())
        expected_dictionary = pa.array(['', 'a', 'bb', 'ccc'], type=value_type)
        assert a.indices.equals(expected_indices)
        assert a.dictionary.equals(expected_dictionary)
    typ = pa.dictionary(pa.int8(), pa.binary(3))
    a = pa.array(['aaa', 'aaa', 'bbb', 'ccc', 'bbb'], type=typ)
    assert isinstance(a.type, pa.DictionaryType)
    expected_indices = pa.array([0, 0, 1, 2, 1], type=pa.int8())
    expected_dictionary = pa.array(['aaa', 'bbb', 'ccc'], type=pa.binary(3))
    assert a.indices.equals(expected_indices)
    assert a.dictionary.equals(expected_dictionary)