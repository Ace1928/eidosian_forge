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
def test_map_type():
    ty = pa.map_(pa.utf8(), pa.int32())
    assert isinstance(ty, pa.MapType)
    assert ty.key_type == pa.utf8()
    assert ty.key_field == pa.field('key', pa.utf8(), nullable=False)
    assert ty.item_type == pa.int32()
    assert ty.item_field == pa.field('value', pa.int32(), nullable=True)
    ty_non_nullable = pa.map_(pa.utf8(), pa.field('value', pa.int32(), nullable=False))
    assert ty != ty_non_nullable
    ty_named = pa.map_(pa.field('x', pa.utf8(), nullable=False), pa.field('y', pa.int32()))
    assert ty == ty_named
    assert not ty.equals(ty_named, check_metadata=True)
    ty_metadata = pa.map_(pa.utf8(), pa.field('value', pa.int32(), metadata={'hello': 'world'}))
    assert ty == ty_metadata
    assert not ty.equals(ty_metadata, check_metadata=True)
    for keys_sorted in [True, False]:
        assert pa.map_(pa.utf8(), pa.int32(), keys_sorted=keys_sorted).keys_sorted == keys_sorted
    with pytest.raises(TypeError):
        pa.map_(None)
    with pytest.raises(TypeError):
        pa.map_(pa.int32(), None)
    with pytest.raises(TypeError):
        pa.map_(pa.field('name', pa.string(), nullable=True), pa.int64())