import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_main_sequence_begin_end_items_post(self):
    round_trip('\n        # C start a\n        # C start b\n        - abc      # abc comment\n        - ghi\n        - klm      # klm comment\n        # C end a\n        # C end b\n        ')