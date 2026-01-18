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
def test_is_integer_value():
    assert pa.types.is_integer_value(1)
    assert pa.types.is_integer_value(np.int64(1))
    assert not pa.types.is_integer_value('1')