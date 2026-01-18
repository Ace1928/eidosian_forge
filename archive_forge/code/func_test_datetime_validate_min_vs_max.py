import pytest
from contextlib import nullcontext
import datetime
import itertools
import pytz
from traitlets import TraitError
from ..widget_datetime import DatetimePicker
def test_datetime_validate_min_vs_max():
    dt = dt_2002
    dt_min = datetime.datetime(2112, 1, 1, tzinfo=pytz.utc)
    dt_max = dt_2056
    w = DatetimePicker(value=dt, max=dt_max)
    with pytest.raises(TraitError):
        w.min = dt_min