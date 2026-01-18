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
def test_decimal_properties():
    ty = pa.decimal128(19, 4)
    assert ty.byte_width == 16
    assert ty.precision == 19
    assert ty.scale == 4
    ty = pa.decimal256(76, 38)
    assert ty.byte_width == 32
    assert ty.precision == 76
    assert ty.scale == 38