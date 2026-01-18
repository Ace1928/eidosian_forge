import datetime
import decimal
import pytest
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.tests import util
@pytest.mark.parametrize('typ', [pa.date32(), pa.date64()])
def test_cast_string_to_date(typ):
    scalar = pa.scalar('2021-01-01')
    result = scalar.cast(typ)
    assert result == pa.scalar(datetime.date(2021, 1, 1), type=typ)