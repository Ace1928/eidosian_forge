from collections import OrderedDict
from collections.abc import Iterator
from functools import partial
import datetime
import sys
import pytest
import hypothesis as h
import hypothesis.strategies as st
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.types as types
import pyarrow.tests.strategies as past
def test_is_list():
    a = pa.list_(pa.int32())
    b = pa.large_list(pa.int32())
    c = pa.list_(pa.int32(), 3)
    assert types.is_list(a)
    assert not types.is_large_list(a)
    assert not types.is_fixed_size_list(a)
    assert types.is_large_list(b)
    assert not types.is_list(b)
    assert not types.is_fixed_size_list(b)
    assert types.is_fixed_size_list(c)
    assert not types.is_list(c)
    assert not types.is_large_list(c)
    assert not types.is_list(pa.int32())