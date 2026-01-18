import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
@property
def ins(self):
    return '        first_name: Art\n        occupation: Architect  # This is an occupation comment\n        about: Art Vandelay is a fictional character that George invents...\n        '