from __future__ import print_function
import sys
import pytest  # NOQA
import platform
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_blank_line_between_seq_items(self):
    round_trip('\n        # Seq with empty lines in between items.\n        b:\n        - bar\n\n\n        - baz\n        ')