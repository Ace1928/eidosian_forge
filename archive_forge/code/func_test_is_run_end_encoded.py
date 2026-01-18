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
def test_is_run_end_encoded():
    assert types.is_run_end_encoded(pa.run_end_encoded(pa.int32(), pa.int64()))
    assert not types.is_run_end_encoded(pa.utf8())