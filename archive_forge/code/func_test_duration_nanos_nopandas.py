import datetime
import decimal
import pytest
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.tests import util
@pytest.mark.nopandas
def test_duration_nanos_nopandas():
    arr = pa.array([0, 3600000000000], pa.duration('ns'))
    expected = datetime.timedelta(seconds=60 * 60)
    assert isinstance(arr[1].as_py(), datetime.timedelta)
    assert arr[1].as_py() == expected
    assert arr[1].value == expected.total_seconds() * 1000000000.0
    arr = pa.array([946684800000000001], type=pa.duration('ns'))
    with pytest.raises(ValueError):
        arr[0].as_py()