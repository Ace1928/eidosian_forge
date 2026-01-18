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
def test_cast_from_null():
    in_data = [None] * 3
    in_type = pa.null()
    out_types = [pa.null(), pa.uint8(), pa.float16(), pa.utf8(), pa.binary(), pa.binary(10), pa.list_(pa.int16()), pa.list_(pa.int32(), 4), pa.large_list(pa.uint8()), pa.decimal128(19, 4), pa.timestamp('us'), pa.timestamp('us', tz='UTC'), pa.timestamp('us', tz='Europe/Paris'), pa.duration('us'), pa.month_day_nano_interval(), pa.struct([pa.field('a', pa.int32()), pa.field('b', pa.list_(pa.int8())), pa.field('c', pa.string())]), pa.dictionary(pa.int32(), pa.string())]
    for out_type in out_types:
        _check_cast_case((in_data, in_type, in_data, out_type))
    out_types = [pa.union([pa.field('a', pa.binary(10)), pa.field('b', pa.string())], mode=pa.lib.UnionMode_DENSE), pa.union([pa.field('a', pa.binary(10)), pa.field('b', pa.string())], mode=pa.lib.UnionMode_SPARSE)]
    in_arr = pa.array(in_data, type=pa.null())
    for out_type in out_types:
        with pytest.raises(NotImplementedError):
            in_arr.cast(out_type)