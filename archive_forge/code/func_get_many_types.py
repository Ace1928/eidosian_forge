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
def get_many_types():
    return (pa.null(), pa.bool_(), pa.int32(), pa.time32('s'), pa.time64('us'), pa.date32(), pa.timestamp('us'), pa.timestamp('us', tz='UTC'), pa.timestamp('us', tz='Europe/Paris'), pa.duration('s'), pa.float16(), pa.float32(), pa.float64(), pa.decimal128(19, 4), pa.decimal256(76, 38), pa.string(), pa.binary(), pa.binary(10), pa.large_string(), pa.large_binary(), pa.list_(pa.int32()), pa.list_(pa.int32(), 2), pa.large_list(pa.uint16()), pa.map_(pa.string(), pa.int32()), pa.map_(pa.field('key', pa.int32(), nullable=False), pa.field('value', pa.int32())), pa.struct([pa.field('a', pa.int32()), pa.field('b', pa.int8()), pa.field('c', pa.string())]), pa.struct([pa.field('a', pa.int32(), nullable=False), pa.field('b', pa.int8(), nullable=False), pa.field('c', pa.string())]), pa.union([pa.field('a', pa.binary(10)), pa.field('b', pa.string())], mode=pa.lib.UnionMode_DENSE), pa.union([pa.field('a', pa.binary(10)), pa.field('b', pa.string())], mode=pa.lib.UnionMode_DENSE, type_codes=[4, 8]), pa.union([pa.field('a', pa.binary(10)), pa.field('b', pa.string())], mode=pa.lib.UnionMode_SPARSE), pa.union([pa.field('a', pa.binary(10), nullable=False), pa.field('b', pa.string())], mode=pa.lib.UnionMode_SPARSE), pa.dictionary(pa.int32(), pa.string()), pa.run_end_encoded(pa.int16(), pa.int32()), pa.run_end_encoded(pa.int32(), pa.string()), pa.run_end_encoded(pa.int64(), pa.uint8()))