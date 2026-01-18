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
def test_decimal_to_decimal():
    arr = pa.array([decimal.Decimal('1234.12'), None], type=pa.decimal128(19, 10))
    result = arr.cast(pa.decimal128(15, 6))
    expected = pa.array([decimal.Decimal('1234.12'), None], type=pa.decimal128(15, 6))
    assert result.equals(expected)
    msg_regexp = 'Rescaling Decimal128 value would cause data loss'
    with pytest.raises(pa.ArrowInvalid, match=msg_regexp):
        result = arr.cast(pa.decimal128(9, 1))
    result = arr.cast(pa.decimal128(9, 1), safe=False)
    expected = pa.array([decimal.Decimal('1234.1'), None], type=pa.decimal128(9, 1))
    assert result.equals(expected)
    with pytest.raises(pa.ArrowInvalid, match='Decimal value does not fit in precision'):
        result = arr.cast(pa.decimal128(5, 2))