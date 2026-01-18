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
def test_dictionary_ordered_equals():
    d1 = pa.dictionary('int32', 'binary', ordered=True)
    d2 = pa.dictionary('int32', 'binary', ordered=False)
    d3 = pa.dictionary('int8', 'binary', ordered=True)
    d4 = pa.dictionary('int32', 'binary', ordered=True)
    assert not d1.equals(d2)
    assert not d1.equals(d3)
    assert d1.equals(d4)