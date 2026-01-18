import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_dropped(self):
    s = '        # comment\n        scalar\n        ...\n        '
    round_trip(s, 'scalar\n...\n')