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
def test_field_metadata():
    f1 = pa.field('a', pa.int8())
    f2 = pa.field('a', pa.int8(), metadata={})
    f3 = pa.field('a', pa.int8(), metadata={b'bizz': b'bazz'})
    assert f1.metadata is None
    assert f2.metadata == {}
    assert f3.metadata[b'bizz'] == b'bazz'