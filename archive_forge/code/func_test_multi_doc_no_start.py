import pytest  # NOQA
from .roundtrip import round_trip, round_trip_load_all
def test_multi_doc_no_start(self):
    inp = '        - a\n        ...\n        ---\n        - b\n        ...\n        '
    docs = list(round_trip_load_all(inp))
    assert docs == [['a'], ['b']]