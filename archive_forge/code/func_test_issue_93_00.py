import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_issue_93_00(self):
    round_trip('        a:\n        - - c1: cat   # a1\n          # my comment on catfish\n          - c2: catfish  # a2\n        ')