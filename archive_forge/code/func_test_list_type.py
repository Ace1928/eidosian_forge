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
def test_list_type():
    ty = pa.list_(pa.int64())
    assert isinstance(ty, pa.ListType)
    assert ty.value_type == pa.int64()
    assert ty.value_field == pa.field('item', pa.int64(), nullable=True)
    ty_non_nullable = pa.list_(pa.field('item', pa.int64(), nullable=False))
    assert ty != ty_non_nullable
    ty_named = pa.list_(pa.field('element', pa.int64()))
    assert ty == ty_named
    assert not ty.equals(ty_named, check_metadata=True)
    ty_metadata = pa.list_(pa.field('item', pa.int64(), metadata={'hello': 'world'}))
    assert ty == ty_metadata
    assert not ty.equals(ty_metadata, check_metadata=True)
    with pytest.raises(TypeError):
        pa.list_(None)