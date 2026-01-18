import datetime
import decimal
import pytest
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.tests import util
def test_decimal128():
    v = decimal.Decimal('1.123')
    s = pa.scalar(v)
    assert isinstance(s, pa.Decimal128Scalar)
    assert s.as_py() == v
    assert s.type == pa.decimal128(4, 3)
    v = decimal.Decimal('1.1234')
    with pytest.raises(pa.ArrowInvalid):
        pa.scalar(v, type=pa.decimal128(4, scale=3))
    with pytest.raises(pa.ArrowInvalid):
        pa.scalar(v, type=pa.decimal128(5, scale=3))
    s = pa.scalar(v, type=pa.decimal128(5, scale=4))
    assert isinstance(s, pa.Decimal128Scalar)
    assert s.as_py() == v