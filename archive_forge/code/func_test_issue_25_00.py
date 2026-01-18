import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_issue_25_00(self):
    round_trip('        params:\n          a: 1 # comment a\n          b:   # comment b\n        ')