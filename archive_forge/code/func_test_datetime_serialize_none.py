import pytest
import datetime
import pytz
from traitlets import TraitError
from ..trait_types import (
def test_datetime_serialize_none():
    assert datetime_to_json(None, None) == None