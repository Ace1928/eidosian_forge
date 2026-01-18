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
def test_decimal_to_int_value_out_of_bounds():
    out_of_bounds_cases = [(np.array([decimal.Decimal('1234567890123'), None, decimal.Decimal('-912345678901234')]), pa.decimal128(32, 5), [1912276171, None, -135950322], pa.int32()), ([decimal.Decimal('123456'), None, decimal.Decimal('-912345678')], pa.decimal128(32, 5), [-7616, None, -19022], pa.int16()), ([decimal.Decimal('1234'), None, decimal.Decimal('-9123')], pa.decimal128(32, 5), [-46, None, 93], pa.int8())]
    for case in out_of_bounds_cases:
        with pytest.raises(pa.ArrowInvalid, match='Integer value out of bounds'):
            _check_cast_case(case)
        _check_cast_case(case, safe=False, check_array_construction=False)