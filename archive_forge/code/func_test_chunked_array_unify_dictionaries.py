from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_chunked_array_unify_dictionaries():
    arr = pa.chunked_array([pa.array(['foo', 'bar', None, 'foo']).dictionary_encode(), pa.array(['quux', None, 'foo']).dictionary_encode()])
    assert arr.chunk(0).dictionary.equals(pa.array(['foo', 'bar']))
    assert arr.chunk(1).dictionary.equals(pa.array(['quux', 'foo']))
    arr = arr.unify_dictionaries()
    expected_dict = pa.array(['foo', 'bar', 'quux'])
    assert arr.chunk(0).dictionary.equals(expected_dict)
    assert arr.chunk(1).dictionary.equals(expected_dict)
    assert arr.to_pylist() == ['foo', 'bar', None, 'foo', 'quux', None, 'foo']