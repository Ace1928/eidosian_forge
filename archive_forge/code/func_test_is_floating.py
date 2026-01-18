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
def test_is_floating():
    for t in [pa.float16(), pa.float32(), pa.float64()]:
        assert types.is_floating(t)
    assert not types.is_floating(pa.int32())