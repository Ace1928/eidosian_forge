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
def test_field_equality_operators():
    f1 = pa.field('a', pa.int8(), nullable=True)
    f2 = pa.field('a', pa.int8(), nullable=True)
    f3 = pa.field('b', pa.int8(), nullable=True)
    f4 = pa.field('b', pa.int8(), nullable=False)
    assert f1 == f2
    assert f1 != f3
    assert f3 != f4
    assert f1 != 'foo'