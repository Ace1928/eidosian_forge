import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_no_comments(self):
    round_trip('\n        - europe: 10\n        - usa:\n          - ohio: 2\n          - california: 9\n        ')