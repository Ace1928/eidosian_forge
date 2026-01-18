import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_insert_1(self):
    d = round_trip_load(self.ins)
    d['ab'].insert(4, 'xyz')
    y = round_trip_dump(d, indent=2)
    assert y == dedent('        ab:\n        - a      # a\n        - b      # b\n        - c\n        - d      # d\n\n        - xyz\n        de:\n        - 1\n        - 2\n        ')