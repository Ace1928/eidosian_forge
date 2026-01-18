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
def test_field_basic():
    t = pa.string()
    f = pa.field('foo', t)
    assert f.name == 'foo'
    assert f.nullable
    assert f.type is t
    assert repr(f) == 'pyarrow.Field<foo: string>'
    f = pa.field('foo', t, False)
    assert not f.nullable
    with pytest.raises(TypeError):
        pa.field('foo', None)