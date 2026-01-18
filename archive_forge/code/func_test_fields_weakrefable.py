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
def test_fields_weakrefable():
    field = pa.field('a', pa.int32())
    wr = weakref.ref(field)
    assert wr() is not None
    del field
    assert wr() is None