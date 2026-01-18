import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_insert_0(self):
    d = round_trip_load(self.ins)
    d['ab'].insert(0, 'xyz')
    y = round_trip_dump(d, indent=2)
    assert y == dedent('        ab:\n        - xyz\n        - a      # a\n        - b      # b\n        - c\n        - d      # d\n\n        de:\n        - 1\n        - 2\n        ')