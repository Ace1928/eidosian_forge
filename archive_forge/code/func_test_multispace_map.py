import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_multispace_map(self):
    round_trip('\n        a: 1x\n\n        b: 2x\n\n\n        c: 3x\n\n\n\n        d: 4x\n\n        ')