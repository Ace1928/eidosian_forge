import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_issue_54_ok(self):
    yaml_str = dedent('        toplevel:\n            # some comment\n            sublevel: 300\n        ')
    d = round_trip_load(yaml_str)
    y = round_trip_dump(d, indent=4)
    assert yaml_str == y