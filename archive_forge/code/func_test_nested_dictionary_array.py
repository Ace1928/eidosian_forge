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
def test_nested_dictionary_array():
    dict_arr = pa.DictionaryArray.from_arrays([0, 1, 0], ['a', 'b'])
    list_arr = pa.ListArray.from_arrays([0, 2, 3], dict_arr)
    assert list_arr.to_pylist() == [['a', 'b'], ['a']]
    dict_arr = pa.DictionaryArray.from_arrays([0, 1, 0], ['a', 'b'])
    dict_arr2 = pa.DictionaryArray.from_arrays([0, 1, 2, 1, 0], dict_arr)
    assert dict_arr2.to_pylist() == ['a', 'b', 'a', 'b', 'a']