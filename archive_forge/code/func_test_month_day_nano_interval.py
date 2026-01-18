import datetime
import decimal
import pytest
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.tests import util
def test_month_day_nano_interval():
    triple = pa.MonthDayNano([-3600, 1800, -50])
    arr = pa.array([triple])
    assert isinstance(arr[0].as_py(), pa.MonthDayNano)
    assert arr[0].as_py() == triple
    assert arr[0].value == triple