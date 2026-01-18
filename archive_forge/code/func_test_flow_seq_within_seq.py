import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_flow_seq_within_seq(self):
    round_trip('        # comment 1\n        - a\n        - b\n        # comment 2\n        - c\n        - d\n        # comment 3\n        - [e]\n        - f\n        # comment 4\n        - []\n        ')