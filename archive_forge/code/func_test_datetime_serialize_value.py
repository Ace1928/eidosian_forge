import pytest
import datetime
import pytz
from traitlets import TraitError
from ..trait_types import (
def test_datetime_serialize_value():
    t = datetime.datetime(2002, 2, 20, 13, 37, 42, 7000, pytz.utc)
    assert datetime_to_json(t, None) == dict(year=2002, month=1, date=20, hours=13, minutes=37, seconds=42, milliseconds=7)