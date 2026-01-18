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
def test_decimal_overflow():
    pa.decimal128(1, 0)
    pa.decimal128(38, 0)
    for i in (0, -1, 39):
        with pytest.raises(ValueError):
            pa.decimal128(i, 0)
    pa.decimal256(1, 0)
    pa.decimal256(76, 0)
    for i in (0, -1, 77):
        with pytest.raises(ValueError):
            pa.decimal256(i, 0)