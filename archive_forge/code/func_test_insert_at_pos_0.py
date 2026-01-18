import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_insert_at_pos_0(self):
    d = round_trip_load(self.ins)
    d.insert(0, 'last name', 'Vandelay', comment='new key')
    y = round_trip_dump(d)
    print(y)
    assert y == dedent('        last name: Vandelay  # new key\n        first_name: Art\n        occupation: Architect  # This is an occupation comment\n        about: Art Vandelay is a fictional character that George invents...\n        ')