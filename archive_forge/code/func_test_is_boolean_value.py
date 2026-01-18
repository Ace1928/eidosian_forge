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
def test_is_boolean_value():
    assert not pa.types.is_boolean_value(1)
    assert pa.types.is_boolean_value(True)
    assert pa.types.is_boolean_value(False)
    assert pa.types.is_boolean_value(np.bool_(True))
    assert pa.types.is_boolean_value(np.bool_(False))