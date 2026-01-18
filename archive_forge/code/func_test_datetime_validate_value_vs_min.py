import pytest
from contextlib import nullcontext
import datetime
import itertools
import pytz
from traitlets import TraitError
from ..widget_datetime import DatetimePicker
def test_datetime_validate_value_vs_min():
    dt = dt_2002
    dt_min = datetime.datetime(2019, 1, 1, tzinfo=pytz.utc)
    dt_max = dt_2056
    w = DatetimePicker(min=dt_min, max=dt_max)
    w.value = dt
    assert w.value.year == 2019