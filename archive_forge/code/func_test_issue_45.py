import copy
import pytest  # NOQA
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_issue_45(self):
    round_trip('\n        dt: 2016-08-19T22:45:47Z\n        ')