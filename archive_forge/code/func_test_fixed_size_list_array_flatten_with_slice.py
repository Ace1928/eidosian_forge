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
def test_fixed_size_list_array_flatten_with_slice():
    array = pa.array([[1], [2], [3]], type=pa.list_(pa.float64(), list_size=1))
    assert array[2:].flatten() == pa.array([3], type=pa.float64())