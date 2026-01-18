from collections.abc import Iterable
import datetime
import decimal
import hypothesis as h
import hypothesis.strategies as st
import itertools
import pytest
import struct
import subprocess
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.tests.strategies as past
def test_empty_cast():
    types = [pa.null(), pa.bool_(), pa.int8(), pa.int16(), pa.int32(), pa.int64(), pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64(), pa.float16(), pa.float32(), pa.float64(), pa.date32(), pa.date64(), pa.binary(), pa.binary(length=4), pa.string()]
    for t1, t2 in itertools.product(types, types):
        try:
            pa.array([], type=t1).cast(t2)
        except (pa.lib.ArrowNotImplementedError, pa.ArrowInvalid):
            continue