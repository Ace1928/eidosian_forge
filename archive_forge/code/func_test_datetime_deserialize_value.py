import pytest
import datetime
import pytz
from traitlets import TraitError
from ..trait_types import (
def test_datetime_deserialize_value():
    tz = pytz.FixedOffset(42)
    v = dict(year=2002, month=1, date=20, hours=13, minutes=37, seconds=42, milliseconds=7)
    assert datetime_from_json(v, None) == datetime.datetime(2002, 2, 20, 14, 19, 42, 7000, tz)