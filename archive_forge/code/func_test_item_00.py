import pytest  # NOQA
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_item_00(self):
    data = load('\n            - a\n            - e\n            - [b, d]\n            - c\n            ')
    assert data[2].lc.line == 2
    assert data[2].lc.col == 2