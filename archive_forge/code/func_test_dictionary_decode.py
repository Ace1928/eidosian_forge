from collections.abc import Iterable
import datetime
import decimal
import hypothesis as h
import hypothesis.strategies as st
import itertools
import pytest
import struct
import subprocess
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.tests.strategies as past
def test_dictionary_decode():
    cases = [(pa.array([1, 2, 3, None, 1, 2, 3]), pa.DictionaryArray.from_arrays(pa.array([0, 1, 2, None, 0, 1, 2], type='int32'), [1, 2, 3])), (pa.array(['foo', None, 'bar', 'foo']), pa.DictionaryArray.from_arrays(pa.array([0, None, 1, 0], type='int32'), ['foo', 'bar'])), (pa.array(['foo', None, 'bar', 'foo'], type=pa.large_binary()), pa.DictionaryArray.from_arrays(pa.array([0, None, 1, 0], type='int32'), pa.array(['foo', 'bar'], type=pa.large_binary())))]
    for expected, arr in cases:
        result = arr.dictionary_decode()
        assert result.equals(expected)