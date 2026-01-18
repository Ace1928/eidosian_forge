import pytest
from contextlib import nullcontext
import datetime
import itertools
import pytz
from traitlets import TraitError
from ..widget_datetime import DatetimePicker
def test_datetime_validate_value_vs_max():
    dt = dt_2002
    dt_min = dt_1664
    dt_max = dt_1994
    w = DatetimePicker(min=dt_min, max=dt_max)
    w.value = dt
    assert w.value.year == 1994