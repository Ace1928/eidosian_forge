import pytest  # NOQA
from .roundtrip import dedent, round_trip, round_trip_load
def test_load_version_1_1(self):
    inp = '        - 12:34:56\n        - 12:34:56.78\n        - 012\n        - 012345678\n        - 0o12\n        - on\n        - off\n        - yes\n        - no\n        - true\n        '
    r = load(inp, version='1.1')
    assert r[0] == 45296
    assert r[1] == 45296.78
    assert r[2] == 10
    assert r[3] == '012345678'
    assert r[4] == '0o12'
    assert r[5] is True
    assert r[6] is False
    assert r[7] is True
    assert r[8] is False
    assert r[9] is True