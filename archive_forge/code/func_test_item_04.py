import pytest  # NOQA
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_item_04(self):
    data = load('\n         # testing line and column based on SO\n         # http://stackoverflow.com/questions/13319067/\n         - key1: item 1\n           key2: item 2\n         - key3: another item 1\n           key4: another item 2\n            ')
    assert data[0].lc.line == 2
    assert data[0].lc.col == 2
    assert data[1].lc.line == 4
    assert data[1].lc.col == 2