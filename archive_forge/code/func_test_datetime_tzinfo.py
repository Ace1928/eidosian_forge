import pytest
import datetime
import pytz
from traitlets import TraitError
from ..widget_datetime import NaiveDatetimePicker
def test_datetime_tzinfo():
    tz = pytz.timezone('Australia/Sydney')
    t = datetime.datetime(2002, 2, 20, 13, 37, 42, 7, tzinfo=tz)
    with pytest.raises(TraitError):
        w = NaiveDatetimePicker(value=t)