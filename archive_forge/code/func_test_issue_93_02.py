import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_issue_93_02(self):
    round_trip('        - c1: cat\n        # my comment on catfish\n        - c2: catfish\n        ')