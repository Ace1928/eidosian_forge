import pytest
from contextlib import nullcontext
import datetime
import itertools
import pytz
from traitlets import TraitError
from ..widget_datetime import DatetimePicker
def test_datetime_validate_value_none():
    dt = dt_2002
    dt_min = dt_1442
    dt_max = dt_2056
    w = DatetimePicker(value=dt, min=dt_min, max=dt_max)
    w.value = None
    assert w.value is None