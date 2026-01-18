import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_09a(self):
    round_trip('\n        hr: # 1998 hr ranking\n        - Mark McGwire\n        - Sammy Sosa\n        rbi:\n          # 1998 rbi ranking\n        - Sammy Sosa\n        - Ken Griffey\n        ')