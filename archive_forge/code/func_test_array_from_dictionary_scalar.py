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
def test_array_from_dictionary_scalar():
    dictionary = ['foo', 'bar', 'baz']
    arr = pa.DictionaryArray.from_arrays([2, 1, 2, 0], dictionary=dictionary)
    result = pa.repeat(arr[0], 5)
    expected = pa.DictionaryArray.from_arrays([2] * 5, dictionary=dictionary)
    assert result.equals(expected)
    result = pa.repeat(arr[3], 5)
    expected = pa.DictionaryArray.from_arrays([0] * 5, dictionary=dictionary)
    assert result.equals(expected)