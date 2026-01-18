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
def test_is_nested_or_struct():
    struct_ex = pa.struct([pa.field('a', pa.int32()), pa.field('b', pa.int8()), pa.field('c', pa.string())])
    assert types.is_struct(struct_ex)
    assert not types.is_struct(pa.list_(pa.int32()))
    assert types.is_nested(struct_ex)
    assert types.is_nested(pa.list_(pa.int32()))
    assert types.is_nested(pa.large_list(pa.int32()))
    assert not types.is_nested(pa.int32())