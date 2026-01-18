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
@h.given(st.lists(past.all_types) | st.lists(past.all_fields) | st.lists(past.all_schemas))
def test_hashing(items):
    h.assume(all((not a.equals(b) for i, a in enumerate(items) for b in items[:i])))
    container = {}
    for i, item in enumerate(items):
        assert hash(item) == hash(item)
        container[item] = i
    assert len(container) == len(items)
    for i, item in enumerate(items):
        assert container[item] == i