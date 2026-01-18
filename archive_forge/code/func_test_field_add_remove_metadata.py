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
def test_field_add_remove_metadata():
    import collections
    f0 = pa.field('foo', pa.int32())
    assert f0.metadata is None
    metadata = {b'foo': b'bar', b'pandas': b'badger'}
    metadata2 = collections.OrderedDict([(b'a', b'alpha'), (b'b', b'beta')])
    f1 = f0.with_metadata(metadata)
    assert f1.metadata == metadata
    f2 = f0.with_metadata(metadata2)
    assert f2.metadata == metadata2
    with pytest.raises(TypeError):
        f0.with_metadata([1, 2, 3])
    f3 = f1.remove_metadata()
    assert f3.metadata is None
    f4 = f3.remove_metadata()
    assert f4.metadata is None
    f5 = pa.field('foo', pa.int32(), True, metadata)
    f6 = f0.with_metadata(metadata)
    assert f5.equals(f6)