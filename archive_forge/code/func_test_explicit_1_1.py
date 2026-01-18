import pytest  # NOQA
from .roundtrip import dedent, round_trip, round_trip_load
def test_explicit_1_1(self):
    r = load('        %YAML 1.1\n        ---\n        - 12:34:56\n        - 012\n        - 012345678\n        - 0o12\n        - on\n        - off\n        - yes\n        - no\n        - true\n        ')
    assert r[0] == 45296
    assert r[1] == 10
    assert r[2] == '012345678'
    assert r[3] == '0o12'
    assert r[4] is True
    assert r[5] is False
    assert r[6] is True
    assert r[7] is False
    assert r[8] is True