import datetime
import decimal
import pytest
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.tests import util
@pytest.mark.pandas
def test_duration_nanos_pandas():
    import pandas as pd
    arr = pa.array([0, 3600000000000], type=pa.duration('ns'))
    expected = pd.Timedelta('1 hour')
    assert isinstance(arr[1].as_py(), pd.Timedelta)
    assert arr[1].as_py() == expected
    assert arr[1].value == expected.value
    arr = pa.array([946684800000000001], type=pa.duration('ns'))
    assert arr[0].as_py() == pd.Timedelta(946684800000000001, unit='ns')